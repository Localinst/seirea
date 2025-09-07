#!/usr/bin/env python3
"""Predict a single match (home vs away on date), return top probability, fair odds,
and the maximum bookmaker odds available in partite/I*.csv files.

Also indicate whether the match would NOT be eligible under simulate_betting rules
(threshold=1.25, min_fair=1.4).

Usage:
  python tools/predict_with_bookies.py --home Inter --away Monza --date 2023-08-19
"""
import os
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import importlib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts_2'


def normalize_team(s: str) -> str:
    import unicodedata, re
    if pd.isna(s):
        return ''
    s = str(s).strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r'[^0-9a-z ]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def find_match_row(Date, Home, Away, partite_dir: Path):
    # read all I*.csv and concat
    files = sorted(partite_dir.glob('I*.csv'))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, dayfirst=True)
            d['source_file'] = f.name
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return None
    allp = pd.concat(dfs, ignore_index=True)
    # robust date parsing
    allp['Date_parsed'] = pd.to_datetime(allp['Date'], dayfirst=True, errors='coerce')
    mask_na = allp['Date_parsed'].isna()
    if mask_na.any():
        def try_alt(x):
            for fmt in ('%d/%m/%y','%d-%m-%y','%Y-%m-%d'):
                try:
                    return pd.to_datetime(x, format=fmt, dayfirst=True)
                except Exception:
                    continue
            return pd.NaT
        allp.loc[mask_na, 'Date_parsed'] = allp.loc[mask_na, 'Date'].apply(try_alt)
    allp['Date_only'] = allp['Date_parsed'].dt.date

    # try exact match first (case-insensitive)
    date_obj = pd.to_datetime(Date).date()
    mask = (allp['Date_only'] == date_obj) & (allp['HomeTeam'].str.lower() == str(Home).lower()) & (allp['AwayTeam'].str.lower() == str(Away).lower())
    if mask.any():
        return allp[mask].iloc[0]

    # try normalized name match and date +/-1 day
    home_n = normalize_team(Home)
    away_n = normalize_team(Away)
    allp['Home_norm'] = allp['HomeTeam'].apply(normalize_team)
    allp['Away_norm'] = allp['AwayTeam'].apply(normalize_team)
    cand_dates = [date_obj]
    try:
        dt = pd.to_datetime(date_obj)
        cand_dates.extend([(dt - pd.Timedelta(days=1)).date(), (dt + pd.Timedelta(days=1)).date()])
    except Exception:
        pass
    for cd in cand_dates:
        sub = allp[allp['Date_only'] == cd]
        if sub.empty:
            continue
        # find best match by simple equality of normalized names
        mask2 = (sub['Home_norm'] == home_n) & (sub['Away_norm'] == away_n)
        if mask2.any():
            return sub[mask2].iloc[0]
    # fallback: try fuzzy by seq matcher on normalized names
    import difflib
    best_score = 0.0
    best_row = None
    for _, r in allp.iterrows():
        if pd.isna(r.get('Home_norm')) or pd.isna(r.get('Away_norm')):
            continue
        s1 = difflib.SequenceMatcher(None, home_n, r['Home_norm']).ratio()
        s2 = difflib.SequenceMatcher(None, away_n, r['Away_norm']).ratio()
        score = s1 + s2
        if score > best_score:
            best_score = score
            best_row = r
    if best_score >= 1.2:
        return best_row
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--home', required=True)
    p.add_argument('--away', required=True)
    p.add_argument('--date', required=True)
    p.add_argument('--seed-results', required=False)
    args = p.parse_args()

    # import predict_next utility: ensure repo root is on sys.path, fallback to file load
    try:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import predict.predict_next as pn
    except Exception as e:
        # fallback: try to load by file path
        try:
            import importlib.util
            spec_path = ROOT / 'predict' / 'predict_next.py'
            if spec_path.exists():
                spec = importlib.util.spec_from_file_location('predict.predict_next', str(spec_path))
                pn = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pn)
            else:
                print('Could not import predict.predict_next:', e)
                return
        except Exception as e2:
            print('Could not import predict.predict_next:', e2)
            return

    # try to find row in features, else compute on the fly
    FEATS = ROOT / 'data' / 'features.csv'
    if FEATS.exists():
        feats = pd.read_csv(FEATS, parse_dates=['Date'])
        try:
            date_obj = pd.to_datetime(args.date).date()
        except Exception:
            print('Invalid date format. Use YYYY-MM-DD')
            return
        mask = (feats['Date'].dt.date == date_obj) & (feats['HomeTeam'].str.lower() == args.home.lower()) & (feats['AwayTeam'].str.lower() == args.away.lower())
        if mask.any():
            row = feats[mask].iloc[0]
        else:
            # compute on the fly using combined_matches.csv
            COMBINED = ROOT / 'data' / 'combined_matches.csv'
            try:
                rowd = pn.compute_on_the_fly(args.home, args.away, args.date, str(COMBINED), seed_path=args.seed_results)
                row = pd.Series(rowd)
            except Exception as e:
                print('Could not compute features on-the-fly:', e)
                return
    else:
        # no features.csv, compute on-the-fly
        COMBINED = ROOT / 'data' / 'combined_matches.csv'
        try:
            rowd = pn.compute_on_the_fly(args.home, args.away, args.date, str(COMBINED), seed_path=args.seed_results)
            row = pd.Series(rowd)
        except Exception as e:
            print('Could not compute features on-the-fly:', e)
            return

    # load model artifact (prefer calibrated)
    ARTDIR = ART
    stacked_cal = ARTDIR / 'stacked_calibrated_model.joblib'
    stacked = ARTDIR / 'stacked_model.joblib'
    stack_path = stacked_cal if stacked_cal.exists() else stacked
    try:
        model_obj, stored_features = pn.load_artifact(str(stack_path))
    except Exception as e:
        print('Could not load model artifact:', e)
        return

    all_num = []
    try:
        # load global numeric cols from data/features.csv if present
        if FEATS.exists():
            df_feats = pd.read_csv(FEATS)
            all_num = [c for c in df_feats.columns if pd.api.types.is_numeric_dtype(df_feats[c])]
    except Exception:
        all_num = []

    model_feat = pn.resolve_feature_cols_for_model(model_obj, stored_features, all_num)

    # Build X and predict
    if isinstance(row, pd.Series):
        X = pd.DataFrame([row.to_dict()])
    else:
        X = pd.DataFrame([row])
    X = X.reindex(columns=model_feat).fillna(0.0)
    probs = pn.predict_proba_safe(model_obj, X, model_feat)[0]
    pred_idx = int(np.argmax(probs))
    predicted_pct = float(probs[pred_idx])
    # predicted outcome in FTR style: H/D/A
    pred_map = {0: 'H', 1: 'D', 2: 'A'}
    predicted_norm = pred_map.get(pred_idx, None)
    fair_odds = None
    if predicted_pct > 0:
        fair_odds = 1.0 / predicted_pct

    # determine if date is in the future relative to today; if so, don't attempt bookie lookup
    try:
        date_obj = pd.to_datetime(args.date).date()
    except Exception:
        date_obj = None

    partite_dir = ROOT / 'partite'
    match_row = None
    max_bookie_odds = None
    max_bookie = None

    from datetime import date as _date
    today = _date.today()
    future_date = (date_obj is not None and date_obj > today)

    if not future_date:
        match_row = find_match_row(args.date, args.home, args.away, partite_dir)
        # determine max bookmaker odds for predicted side
        if match_row is not None:
            # find columns ending with H/D/A present in that row
            cols = [c for c in match_row.index if isinstance(c, str) and len(c) > 1]
            # candidate bookie cols: those ending with predicted suffix
            suffix = {0: 'H', 1: 'D', 2: 'A'}.get(pred_idx, None)
            book_cols = [c for c in cols if c.endswith(suffix) and c not in ('Home', 'Away', 'Date')]
            # fallback: try known list
            if not book_cols:
                known = ['B365','BF','BWH','IWH','LB','PS','WH','SB','SJ','SY','GB','SO']
                for k in known:
                    cname = k + (suffix or '')
                    if cname in match_row.index:
                        book_cols.append(cname)
            # numeric max
            vals = []
            for c in book_cols:
                try:
                    v = float(match_row.get(c))
                    vals.append((v, c))
                except Exception:
                    continue
            if vals:
                vals = sorted(vals, key=lambda x: (np.nan_to_num(x[0], nan=-1)), reverse=True)
                max_bookie_odds, max_bookie = vals[0]
    else:
        # future/invented match: no bookmaker data available
        match_row = None
        max_bookie_odds = None
        max_bookie = None

    # betting eligibility check (simulate_betting rules)
    threshold = 1.25
    min_fair = 0
    eligible = False
    if (fair_odds is not None) and (max_bookie_odds is not None):
        try:
            eligible = (float(max_bookie_odds) >= float(fair_odds) * threshold) and (float(fair_odds) >= float(min_fair))
        except Exception:
            eligible = False
    else:
        # no bookmaker information -> cannot place a bet
        eligible = False

    result = {
        'date': args.date,
        'home': args.home,
        'away': args.away,
    'predicted': predicted_norm,
    'predicted_pct': round(predicted_pct, 4),
        'fair_odds': round(fair_odds, 4) if fair_odds is not None else None,
        'max_bookie_odds': float(max_bookie_odds) if max_bookie_odds is not None else None,
        'max_bookie': str(max_bookie) if max_bookie is not None else None,
        'not_eligible': (not eligible)
    }

    ART.mkdir(parents=True, exist_ok=True)
    outp = ART / f'single_match_bookie_check_{args.home}_{args.away}_{args.date}.json'
    with open(outp, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))
    # Create plots: probability bar and odds comparison
    try:
        # probability bar
        labs = ['Home win', 'Draw', 'Away win']
        vals = [float(probs[0]) * 100.0, float(probs[1]) * 100.0, float(probs[2]) * 100.0]
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(labs, vals, color=['#2ca02c', '#1f77b4', '#d62728'])
        ax.set_ylim(0, 100)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha='center')
        ax.set_ylabel('Probability (%)')
        ax.set_title(f"Predicted probabilities {args.home} vs {args.away} {args.date}")
        plt.tight_layout()
        prob_fname = ART / f'pred_prob_{args.home}_{args.away}_{args.date}.png'
        fig.savefig(prob_fname)
        plt.close(fig)

        # odds comparison: fair vs max_bookie
        if result.get('fair_odds') is not None and result.get('max_bookie_odds') is not None:
            fo = float(result['fair_odds'])
            mo = float(result['max_bookie_odds'])
            fig, ax = plt.subplots(figsize=(5,3))
            labels = [f'Fair (1/pred)', 'Max bookie']
            values = [fo, mo]
            colors = ['#888888', '#2ca02c' if (not result['not_eligible']) else '#d62728']
            bars = ax.bar(labels, values, color=colors)
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha='center')
            ax.set_ylabel('Odds')
            ax.set_title(f"Fair vs Max bookie odds ({args.home} vs {args.away})")
            # annotate eligibility
            note = 'ELIGIBLE' if (not result['not_eligible']) else 'NOT ELIGIBLE'
            ax.text(0.5, 0.9, note, transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold', color=('green' if (not result['not_eligible']) else 'red'))
            plt.tight_layout()
            odds_fname = ART / f'odds_compare_{args.home}_{args.away}_{args.date}.png'
            fig.savefig(odds_fname)
            plt.close(fig)
    except Exception:
        pass


if __name__ == '__main__':
    main()
