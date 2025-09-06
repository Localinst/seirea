"""Attach target/result columns from combined_matches.csv to data/features.csv.

This script creates a timestamped backup of `data/features.csv`, then merges Result/FTR
or computes the result from FTHG/FTAG if needed. It writes two outputs:
- `data/features_with_target.csv` (safe merged copy)
- overwrites `data/features.csv` with the merged version (so existing training scripts can run)

Run from project root:
  python models\attach_targets.py
"""
import os
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
COMBINED = os.path.join(ROOT, 'data', 'combined_matches.csv')


def compute_result_from_scores(df):
    # expects columns FTHG, FTAG
    res = []
    for _, r in df.iterrows():
        try:
            h = int(r.get('FTHG'))
            a = int(r.get('FTAG'))
        except Exception:
            res.append(None); continue
        if pd.isna(h) or pd.isna(a):
            res.append(None)
        elif h > a:
            res.append('H')
        elif h == a:
            res.append('D')
        else:
            res.append('A')
    return res


def main():
    if not os.path.exists(FEATS):
        print('features.csv not found at', FEATS); return
    if not os.path.exists(COMBINED):
        print('combined_matches.csv not found at', COMBINED); return

    feats = pd.read_csv(FEATS, parse_dates=['Date'])
    comb = pd.read_csv(COMBINED, parse_dates=['Date'])

    # Normalize keys
    for df in (feats, comb):
        if 'HomeTeam' in df.columns: df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip()
        if 'AwayTeam' in df.columns: df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip()

    # pick candidate columns in combined that carry result
    cand_cols = []
    for c in ('Result', 'FTR'):
        if c in comb.columns:
            cand_cols.append(c)
    have_scores = ('FTHG' in comb.columns) and ('FTAG' in comb.columns)

    if not cand_cols and not have_scores:
        print('No Result/FTR or FTHG/FTAG columns found in combined_matches.csv; cannot attach targets.')
        return

    # create a slim combined df with identifiers and result/score
    cols = ['Date', 'HomeTeam', 'AwayTeam']
    for c in ('Result', 'FTR', 'FTHG', 'FTAG'):
        if c in comb.columns and c not in cols:
            cols.append(c)
    comb_slim = comb[cols].copy()

    # If Result/FTR missing but scores present, compute Result
    if 'Result' not in comb_slim.columns:
        if 'FTR' in comb_slim.columns:
            comb_slim['Result'] = comb_slim['FTR']
        elif have_scores:
            comb_slim['Result'] = compute_result_from_scores(comb_slim)

    # Merge (left join features <- combined)
    merged = feats.merge(comb_slim[['Date','HomeTeam','AwayTeam','Result','FTHG','FTAG']], on=['Date','HomeTeam','AwayTeam'], how='left')

    # After merge pandas may have suffixed duplicate columns (e.g. Result_x, Result_y).
    # Normalize so we always have a single 'Result', 'FTHG', 'FTAG' in the merged frame.
    def pick_col(primary, fallback):
        if primary in merged.columns:
            return merged[primary]
        if fallback in merged.columns:
            return merged[fallback]
        return None

    # Result
    res_col = pick_col('Result', 'Result_y')
    if res_col is None:
        res_col = pick_col('Result_x', 'Result')
    if res_col is None and 'FTR' in comb_slim.columns:
        res_col = comb_slim['FTR']
    if res_col is None:
        # try computing from scores if present
        if 'FTHG' in comb_slim.columns and 'FTAG' in comb_slim.columns:
            comb_slim['Result'] = compute_result_from_scores(comb_slim)
            res_col = comb_slim['Result']

    if res_col is not None:
        merged['Result'] = res_col.values

    # For FTHG/FTAG prefer merged values from combined (suffix handling as well)
    for score_col in ('FTHG', 'FTAG'):
        sc = pick_col(score_col, score_col + '_y')
        if sc is None:
            sc = pick_col(score_col + '_x', score_col)
        if sc is not None:
            merged[score_col] = sc.values

    missing = merged['Result'].isna().sum() if 'Result' in merged.columns else merged.shape[0]
    total = merged.shape[0]
    print(f'Merged results: {total-missing}/{total} rows have a Result value; {missing} missing')

    # Backup original features.csv
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bak = FEATS + f'.bak_{ts}'
    os.replace(FEATS, bak)
    print('Backed up original features.csv to', bak)

    # Save merged copy (both a safe copy and overwrite features.csv so training scripts can run)
    out_safe = os.path.join(ROOT, 'data', 'features_with_target.csv')
    merged.to_csv(out_safe, index=False)
    merged.to_csv(FEATS, index=False)
    print('Wrote merged features to', out_safe)
    print('Overwrote', FEATS, 'with Result attached (use the backup if needed)')


if __name__ == '__main__':
    main()
