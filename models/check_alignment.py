import pandas as pd
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sim_path = root / "artifacts" / "simulation_2024_2025_predictions.csv"
    i1_path = root / "partite" / "I1.csv"
    eval_path = root / "artifacts" / "simulation_evaluation.csv"
    report_path = root / "artifacts" / "alignment_report.txt"

    sim = pd.read_csv(sim_path, parse_dates=["Date"]) 
    sim["Date"] = pd.to_datetime(sim["Date"]).dt.date
    i1 = pd.read_csv(i1_path, dayfirst=True)
    i1["Date"] = pd.to_datetime(i1["Date"], dayfirst=True).dt.date
    # normalize time
    try:
        i1["Time"] = pd.to_datetime(i1["Time"], format="%H:%M").dt.time
    except Exception:
        # some Time values might be missing; coerce
        i1["Time"] = pd.to_datetime(i1["Time"], dayfirst=True, errors="coerce").dt.time

    eval_df = pd.read_csv(eval_path, parse_dates=["Date"]) if eval_path.exists() else None
    if eval_df is not None:
        eval_df["Date"] = pd.to_datetime(eval_df["Date"]).dt.date

    key_cols = ["Date", "HomeTeam", "AwayTeam"]

    lines = []
    lines.append(f"Simulation rows: {len(sim)}")
    lines.append(f"I1 rows: {len(i1)}")
    if eval_df is not None:
        lines.append(f"Evaluation rows: {len(eval_df)}")

    # Check duplicates
    sim_dups = sim[sim.duplicated(subset=key_cols, keep=False)].sort_values(key_cols)
    i1_dups = i1[i1.duplicated(subset=key_cols, keep=False)].sort_values(key_cols)
    lines.append(f"Simulation duplicate keys: {len(sim_dups)} rows (unique keys: {sim_dups[key_cols].drop_duplicates().shape[0]})")
    lines.append(f"I1 duplicate keys: {len(i1_dups)} rows (unique keys: {i1_dups[key_cols].drop_duplicates().shape[0]})")

    # For each simulation row, attempt to find matching row(s) in I1
    # build right-side columns without duplicating keys
    right_cols = [c for c in (["Date", "Time"] + key_cols + ["FTR"]) ]
    # ensure uniqueness while preserving order
    seen = set()
    right_cols_unique = []
    for c in right_cols:
        if c not in seen:
            right_cols_unique.append(c)
            seen.add(c)
    merged = pd.merge(sim, i1[right_cols_unique], how="left", on=key_cols, indicator=True)
    missing = merged[merged["_merge"] != "both"]
    lines.append(f"Merged rows with no match in I1: {len(missing)}")
    if len(missing) > 0:
        lines.append("Examples of missing matches (up to 10):")
        lines.extend(missing.head(10)[["Date", "HomeTeam", "AwayTeam", "prob_home", "prob_draw", "prob_away", "predicted"]].astype(str).apply(lambda r: ", ".join(r.values), axis=1).tolist())

    # Check one-to-many matches: group i1 by key and count
    i1_counts = i1.groupby(key_cols).size().reset_index(name="i1_count")
    multi = i1_counts[i1_counts["i1_count"] > 1]
    lines.append(f"I1 keys that appear multiple times: {len(multi)}")
    if len(multi) > 0:
        lines.append("Examples of duplicate keys in I1:")
        lines.extend(multi.head(10).astype(str).apply(lambda r: ", ".join(r.values), axis=1).tolist())

    # Verify predicted matches max probability
    probs = sim[["prob_home", "prob_draw", "prob_away"]]
    max_type = probs.idxmax(axis=1).map({"prob_home": "Home", "prob_draw": "Draw", "prob_away": "Away"})
    mism = sim[sim["predicted"] != max_type]
    lines.append(f"Simulation rows where 'predicted' != max prob: {len(mism)}")
    if len(mism) > 0:
        lines.append("Examples:")
        lines.extend(mism.head(10)[["Date", "HomeTeam", "AwayTeam", "prob_home", "prob_draw", "prob_away", "predicted"]].astype(str).apply(lambda r: ", ".join(r.values), axis=1).tolist())

    # If evaluation exists, check alignment indexes
    if eval_df is not None:
        # rows where merged shows both but fields differ
        # find rows where eval 'predicted' corresponds to sim and FTR exists
        bad = eval_df[eval_df["_merge"] != "both"]
        lines.append(f"Evaluation rows not merged as both: {len(bad)}")
        if len(bad) > 0:
            lines.append("Examples of eval rows with _merge != 'both':")
            lines.extend(bad.head(10)[["Date", "HomeTeam", "AwayTeam", "predicted", "_merge"]].astype(str).apply(lambda r: ", ".join(r.values), axis=1).tolist())

    # Print a few paired examples where mapping occurred
    paired = merged[merged["_merge"] == "both"].head(10)
    if not paired.empty:
        lines.append("\nSample paired rows (simulation row -> matched I1 FTR):")
        for _, r in paired.iterrows():
            lines.append(f"{r['Date']} {r['HomeTeam']} vs {r['AwayTeam']} -> pred: {r.get('predicted')} | FTR: {r.get('FTR')}")

    # Save report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote alignment report to:", report_path)
    print("Summary:\n", "\n".join(lines[:20]))


if __name__ == '__main__':
    main()
