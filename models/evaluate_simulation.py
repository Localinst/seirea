import pandas as pd
import difflib
import re
import unicodedata
import matplotlib.pyplot as plt
from pathlib import Path


# small normalizer used for name-similarity and mapping suggestions
def normalize_str(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^0-9a-z ]+", '', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s


def main():
    root = Path(__file__).resolve().parents[1]
    art = root / "artifacts"
    # try both season files and concat if present
    sim_paths = []
    p1 = art / "simulation_2023_2024_predictions.csv"
    p2 = art / "simulation_2024_2025_predictions.csv"
    if p1.exists():
        sim_paths.append(p1)
    if p2.exists():
        sim_paths.append(p2)
    # fallback to older single file name for backward compatibility
    if not sim_paths:
        p_old = art / "simulation_2024_2025_predictions.csv"
        if p_old.exists():
            sim_paths.append(p_old)
    # read all I*.csv files in partite/ (covers multiple seasons: I1, I2, ...)
    partite_dir = root / "partite"
    i_files = sorted(partite_dir.glob("I*.csv"))
    if not i_files:
        raise SystemExit('No I*.csv files found in partite/')
    out_csv = art / "simulation_evaluation.csv"
    out_png = art / "simulation_evaluation.png"

    # Read simulation(s) and I1 (source of truth time/order)
    if not sim_paths:
        raise SystemExit('No simulation prediction files found in artifacts/')
    sim_list = []
    for sp in sim_paths:
        s = pd.read_csv(sp, parse_dates=["Date"]) 
        # keep origin for debugging
        s['sim_source'] = sp.name
        sim_list.append(s)
    sim = pd.concat(sim_list, ignore_index=True)
    # Normalize Date to date (no time) to match I1 merge behavior
    sim["Date"] = pd.to_datetime(sim["Date"]).dt.date

    # concat all I*.csv into a single authoritative dataframe
    i_list = []
    for ip in i_files:
        try:
            df = pd.read_csv(ip, dayfirst=True)
            df['source_file'] = ip.name
            i_list.append(df)
        except Exception as e:
            print(f"Warning reading {ip.name}: {e}")
    i1 = pd.concat(i_list, ignore_index=True)
    # robust date parsing: try dayfirst parsing, then fallback to two-digit years if needed
    i1["Date"] = pd.to_datetime(i1["Date"], dayfirst=True, errors="coerce")
    # rows that failed to parse, try common alternate formats (e.g., dd/mm/yy)
    mask = i1["Date"].isna()
    if mask.any():
        def try_alt(d):
            try:
                return pd.to_datetime(d, format="%d/%m/%y", dayfirst=True)
            except Exception:
                try:
                    return pd.to_datetime(d, format="%d-%m-%y", dayfirst=True)
                except Exception:
                    return pd.NaT
        i1.loc[mask, "Date"] = i1.loc[mask, "Date"].index.to_series().apply(lambda idx: try_alt(i1.at[idx, 'Date']) )
    i1["Date"] = i1["Date"].dt.date
    # normalize Time from I1
    try:
        i1["Time"] = pd.to_datetime(i1["Time"], format="%H:%M").dt.time
    except Exception:
        i1["Time"] = pd.to_datetime(i1["Time"], errors="coerce").dt.time

    # Attempt to load a canonical team name mapping (if present) to harmonize sim names
    # mapping file should be in partite/team_name_map.csv with columns like sim_name,official_name
    mapping_path = partite_dir / 'team_name_map.csv'
    if mapping_path.exists():
        try:
            map_df = pd.read_csv(mapping_path)
            # expect first two columns to be sim_name, official_name
            cols = list(map_df.columns)
            if len(cols) >= 2:
                sim_col, off_col = cols[0], cols[1]
                mapping = dict(zip(map_df[sim_col], map_df[off_col]))
                # apply mapping to simulation frames (if present)
                sim['HomeTeam'] = sim['HomeTeam'].replace(mapping)
                sim['AwayTeam'] = sim['AwayTeam'].replace(mapping)
                print(f"Applied team-name mapping from {mapping_path.name} to simulation data ({len(mapping)} entries)")
        except Exception as e:
            print(f"Warning reading team mapping {mapping_path.name}: {e}")
    else:
        # if no mapping file, produce suggestions for manual mapping to help the user
        try:
            authority_names = set(i1['HomeTeam'].dropna().unique()).union(set(i1['AwayTeam'].dropna().unique()))
            sim_names = set(sim['HomeTeam'].dropna().unique()).union(set(sim['AwayTeam'].dropna().unique()))
            suggestions = []
            for sname in sorted(sim_names):
                # only suggest when exact match not present
                if sname in authority_names:
                    continue
                best = None
                best_score = 0.0
                for an in authority_names:
                    score = difflib.SequenceMatcher(None, normalize_str(sname), normalize_str(an)).ratio()
                    if score > best_score:
                        best_score = score
                        best = an
                suggestions.append({'sim_name': sname, 'best_match': best, 'score': best_score})
            sugg_df = pd.DataFrame(suggestions).sort_values('score', ascending=False)
            if not sugg_df.empty:
                sugg_path = art / 'team_name_mapping_suggestions.csv'
                sugg_df.to_csv(sugg_path, index=False)
                print(f"Wrote {len(sugg_df)} team-name mapping suggestions to {sugg_path}")
        except Exception:
            pass

    # Ensure predicted is the argmax of probabilities; fix if necessary
    sim = sim.copy()
    probs = sim[["prob_home", "prob_draw", "prob_away"]]
    max_prob = probs.max(axis=1)
    max_type = probs.idxmax(axis=1).map({"prob_home": "Home", "prob_draw": "Draw", "prob_away": "Away"})
    mismatch_rows = sim[sim["predicted"] != max_type]
    if not mismatch_rows.empty:
        print(f"Fixing {len(mismatch_rows)} rows where 'predicted' != max probability")
        sim["predicted"] = max_type
        sim["predicted_pct"] = max_prob

    # Map predicted to FTR style
    map_pred = {"Home": "H", "Away": "A", "Draw": "D"}
    sim["predicted_norm"] = sim["predicted"].map(lambda x: map_pred.get(x, x))

    # Detect bookmaker columns present in I1 and include them in the merge
    expected_book_cols = [
        "B365H","B365D","B365A",
        "BFH","BFD","BFA","BFDH","BFDD","BFDA",
        "BMGMH","BMGMD","BMGMA",
        "BVH","BVD","BVA",
        "BSH","BSD","BSA",
        "BWH","BWD","BWA",
        "CLH","CLD","CLA",
        "GBH","GBD","GBA",
        "IWH","IWD","IWA",
        "LBH","LBD","LBA",
        "PSH","PH","PSD","PD","PSA","PA",
        "SOH","SOD","SOA",
        "SBH","SBD","SBA",
        "SJH","SJD","SJA",
        "SYH","SYD","SYA",
        "WHH","WHD","WHA"
    ]
    bookmaker_cols = [c for c in expected_book_cols if c in i1.columns]
    right_cols = ["Date", "Time", "HomeTeam", "AwayTeam", "FTR"] + bookmaker_cols
    merged = pd.merge(sim, i1[right_cols], how="left", on=["Date", "HomeTeam", "AwayTeam"], indicator=True)

    # If any missing matches, try permissive fallbacks using normalized names and fuzzy matching
    missing = merged[merged["_merge"] != "both"]
    if not missing.empty:
        print(f"Warning: {len(missing)} rows didn't match on exact keys; attempting relaxed/fuzzy fallback")

        def normalize_team(s: str) -> str:
            if pd.isna(s):
                return ""
            s = str(s).strip()
            # remove diacritics
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
            s = s.lower()
            # remove punctuation
            s = re.sub(r"[^0-9a-z ]+", '', s)
            # collapse spaces
            s = re.sub(r"\s+", ' ', s).strip()
            return s

        i1_tmp = i1.copy()
        sim_tmp = sim.copy()
        i1_tmp['Home_norm'] = i1_tmp['HomeTeam'].apply(normalize_team)
        i1_tmp['Away_norm'] = i1_tmp['AwayTeam'].apply(normalize_team)
        sim_tmp['Home_norm'] = sim_tmp['HomeTeam'].apply(normalize_team)
        sim_tmp['Away_norm'] = sim_tmp['AwayTeam'].apply(normalize_team)

        # Try merge on normalized names first
        alt = pd.merge(sim_tmp, i1_tmp[["Date", "Home_norm", "Away_norm", "Time", "FTR"]], how="left", left_on=["Date", "Home_norm", "Away_norm"], right_on=["Date", "Home_norm", "Away_norm"])
        for col in ["Time", "FTR"]:
            merged[col] = merged[col].fillna(alt[col])

        # For any still-missing rows, attempt fuzzy matching of team names within same Date
        still_missing = merged[merged["_merge"] != "both"].copy()
        if not still_missing.empty:
            print(f"Attempting fuzzy matching for {len(still_missing)} remaining rows...")
            # build mapping per date from i1 normalized names to original Time/FTR
            i1_grouped = {}
            for _, row in i1_tmp.iterrows():
                d = row['Date']
                i1_grouped.setdefault(d, []).append(row)

            def fuzzy_fill(row):
                # attempt to find best candidate pair using similarity, allowing date +/-1 day
                d = row['Date']
                home_n = normalize_team(row['HomeTeam'])
                away_n = normalize_team(row['AwayTeam'])
                # collect candidates from date, date-1, date+1
                # consider same date and +/-1 day (use date objects for dict lookup)
                cand_dates = [d]
                try:
                    dt = pd.to_datetime(d)
                    cand_dates.append((dt - pd.Timedelta(days=1)).date())
                    cand_dates.append((dt + pd.Timedelta(days=1)).date())
                except Exception:
                    pass
                best_score = 0.0
                best_row = None
                for cd in cand_dates:
                    cand_list = i1_grouped.get(cd, [])
                    for c in cand_list:
                        # compute similarity for home and away
                        h_sim = difflib.SequenceMatcher(None, home_n, c['Home_norm']).ratio()
                        a_sim = difflib.SequenceMatcher(None, away_n, c['Away_norm']).ratio()
                        score = h_sim + a_sim
                        if score > best_score:
                            best_score = score
                            best_row = c
                # require a reasonable combined score (e.g., avg similarity >= 0.6 -> sum >= 1.2)
                if best_row is not None and best_score >= 1.2:
                    return pd.Series({"Time": best_row.get('Time', pd.NA), "FTR": best_row.get('FTR', pd.NA)})
                # otherwise, fallback to single best home or away match on same date
                cand_list = i1_grouped.get(d, [])
                if cand_list:
                    best_single = None
                    best_single_score = 0.0
                    for c in cand_list:
                        h_sim = difflib.SequenceMatcher(None, home_n, c['Home_norm']).ratio()
                        a_sim = difflib.SequenceMatcher(None, away_n, c['Away_norm']).ratio()
                        if h_sim > best_single_score:
                            best_single_score = h_sim
                            best_single = c
                        if a_sim > best_single_score:
                            best_single_score = a_sim
                            best_single = c
                    if best_single is not None and best_single_score >= 0.7:
                        return pd.Series({"Time": best_single.get('Time', pd.NA), "FTR": best_single.get('FTR', pd.NA)})
                return pd.Series({"Time": pd.NA, "FTR": pd.NA})

            # apply fuzzy_fill to still_missing rows and fill merged
            filled = still_missing.apply(fuzzy_fill, axis=1)
            merged.loc[still_missing.index, 'Time'] = merged.loc[still_missing.index, 'Time'].fillna(filled['Time'])
            merged.loc[still_missing.index, 'FTR'] = merged.loc[still_missing.index, 'FTR'].fillna(filled['FTR'])

        # final count of unresolved
        unresolved = merged[merged["FTR"].isna() | (merged["Time"].isna())]
        if len(unresolved) > 0:
            print(f"After fallbacks, {len(unresolved)} rows still lack Time/FTR; written sample to artifacts/unmatched_sim_rows.csv")
            try:
                unresolved[['Date','HomeTeam','AwayTeam','sim_source']].to_csv(Path(__file__).resolve().parents[1] / 'artifacts' / 'unmatched_sim_rows.csv', index=False)
            except Exception:
                pass

    # Now sort using authoritative Date+Time from I1 when available
    # Build a DateTime_sort by combining Date and Time (if Time exists)
    if "Time" in merged.columns:
        # create DateTime string and parse
        merged['Date_str'] = merged['Date'].astype(str)
        merged['Time_str'] = merged['Time'].astype(str)
        merged['DateTime_sort'] = pd.to_datetime(merged['Date_str'] + ' ' + merged['Time_str'], errors='coerce', dayfirst=True)
        # fallback if parsing failed: use Date only
        merged['DateTime_sort'] = merged['DateTime_sort'].fillna(pd.to_datetime(merged['Date']))
        merged = merged.sort_values(['DateTime_sort', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    else:
        merged = merged.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)

    # compute cumulative metrics on ordered data
    merged["match_index"] = range(1, len(merged) + 1)
    merged["pred_FTR"] = merged["predicted_norm"]
    merged["correct"] = merged["pred_FTR"] == merged["FTR"]
    merged["cum_correct"] = merged["correct"].cumsum()
    merged["cum_accuracy"] = merged["cum_correct"] / merged["match_index"]

    # Compute fair odds from predicted probability (avoid div by zero)
    merged["predicted_pct"] = merged.get("predicted_pct", max_prob)
    merged["fair_odds"] = merged["predicted_pct"].replace({0: pd.NA}).rdiv(1)

    # For each row, find the highest bookmaker odds for the predicted option
    # map predicted_norm (H/D/A) to suffix
    suffix_map = {"H": "H", "D": "D", "A": "A"}
    if bookmaker_cols:
        # prepare per-suffix columns available
        # build dict: {'H': [list of colnames], ...}
        book_cols_by_suffix = {s: [c for c in bookmaker_cols if c.endswith(s)] for s in ["H","D","A"]}

        def row_max_bookie_odds(row):
            s = suffix_map.get(row.get("predicted_norm"), None)
            if not s:
                return pd.Series({"max_bookie_odds": pd.NA, "max_bookie": pd.NA})
            cols = book_cols_by_suffix.get(s, [])
            if not cols:
                return pd.Series({"max_bookie_odds": pd.NA, "max_bookie": pd.NA})
            vals = row[cols]
            # pick numeric max
            try:
                max_val = pd.to_numeric(vals, errors="coerce").max()
            except Exception:
                max_val = pd.NA
            if pd.isna(max_val):
                return pd.Series({"max_bookie_odds": pd.NA, "max_bookie": pd.NA})
            # find first bookmaker that equals max_val
            for col in cols:
                try:
                    v = float(row.get(col))
                except Exception:
                    v = None
                if v == max_val:
                    return pd.Series({"max_bookie_odds": max_val, "max_bookie": col})
            return pd.Series({"max_bookie_odds": max_val, "max_bookie": pd.NA})

        bookie_info = merged.apply(row_max_bookie_odds, axis=1)
        merged = pd.concat([merged, bookie_info], axis=1)
    else:
        merged["max_bookie_odds"] = pd.NA
        merged["max_bookie"] = pd.NA

    total = int(merged["correct"].sum())
    total_matches = len(merged)
    pct_exact = 100.0 * total / total_matches if total_matches else 0.0

    # Save evaluation CSV (ordered)
    merged.to_csv(out_csv, index=False)
    print(f"Wrote evaluation CSV to {out_csv} ({total}/{total_matches} exact = {pct_exact:.2f}%)")

    # Plot cumulative correct and cumulative accuracy side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    ax.step(merged["match_index"], merged["cum_correct"], where="post", color="#2c7fb8", label="Cumulative correct")
    ax.set_xlabel("Match number (chronological)")
    ax.set_ylabel("Cumulative correct predictions")
    ax.set_title("Cumulative correct predictions over the simulated season")
    ax.grid(alpha=0.25)

    
    ax2 = axes[1]
    ax2.axis("off")
    txt = f"Exact predictions:\n{total}/{total_matches}\n{pct_exact:.2f}%"
    ax2.text(0.5, 0.5, txt, fontsize=18, ha="center", va="center", bbox=dict(facecolor="#f0f0f0", boxstyle="round,pad=0.6"))

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wrote evaluation plot to {out_png}")

    # Print first and last 10 rows for quick manual inspection
    print("\nFirst 10 evaluation rows:\n", merged[["Date", "Time", "HomeTeam", "AwayTeam", "predicted", "pred_FTR", "FTR", "correct"]].head(10).to_string(index=False))
    print("\nLast 10 evaluation rows:\n", merged[["Date", "Time", "HomeTeam", "AwayTeam", "predicted", "pred_FTR", "FTR", "correct"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
