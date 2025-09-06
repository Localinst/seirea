import os
import glob
import pandas as pd
import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
PARTITE_DIR = os.path.join(ROOT, 'partite')
OUT = os.path.join(ROOT, 'data', 'combined_matches.csv')


def safe_read(path):
    try:
        return pd.read_csv(path, parse_dates=['Date'], dayfirst=True)
    except Exception:
        # fallback: read without parse, then parse explicitly
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df


def implied_prob(odds):
    # odds -> implied probability, handle 0/NaN
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = 1.0 / odds
    p[~np.isfinite(p)] = np.nan
    return p


def normalize_three(a, b, c):
    arr = np.vstack([a, b, c]).T
    out = np.empty_like(arr)
    out[:] = np.nan
    s = np.nansum(arr, axis=1)
    mask = s > 0
    out[mask] = (arr[mask].T / s[mask]).T
    return out[:, 0], out[:, 1], out[:, 2]


def main():
    files = sorted(glob.glob(os.path.join(PARTITE_DIR, 'I*.csv')))
    if not files:
        print('No I*.csv files found in', PARTITE_DIR)
        return

    parts = []
    for p in files:
        df = safe_read(p)
        if df.empty:
            continue
        df['source_file'] = os.path.basename(p)
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True, sort=False)

    # standardize column names casing (keep as-is but ensure key cols exist)
    # keep a core subset
    core = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
            'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
            'source_file']
    cols_present = [c for c in core if c in combined.columns]

    out = combined[cols_present].copy()

    # list of odds columns we want to consider (common bookies and summaries)
    odds_cols = [
        'B365H', 'B365D', 'B365A',
        'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA',
        'Bb1X2',
        'B365>2.5', 'B365<2.5', 'Avg>2.5', 'Avg<2.5',
        'AHh', 'B365AHH', 'B365AHA'
    ]

    for c in odds_cols:
        if c in combined.columns:
            out[c] = combined[c]

    # compute implied probs for primary 1X2 market: prefer B365 then Avg then Max
    # create numeric arrays
    def get_col_pref(row, names):
        for n in names:
            if n in combined.columns:
                return row[n]
        return np.nan

    # vectorized: pick columns if exist
    use_primary = None
    for pref in (['B365H', 'B365D', 'B365A'], ['AvgH', 'AvgD', 'AvgA'], ['MaxH', 'MaxD', 'MaxA']):
        if all([p in combined.columns for p in pref]):
            use_primary = pref
            break

    if use_primary is not None:
        a = implied_prob(combined[use_primary[0]].values)
        b = implied_prob(combined[use_primary[1]].values)
        c = implied_prob(combined[use_primary[2]].values)
        h, d, a_ = normalize_three(a, b, c)
        out['imp_H'] = h
        out['imp_D'] = d
        out['imp_A'] = a_
    else:
        # try to compute per-row from available columns
        out['imp_H'] = implied_prob(combined.get('B365H', np.nan))
        out['imp_D'] = implied_prob(combined.get('B365D', np.nan))
        out['imp_A'] = implied_prob(combined.get('B365A', np.nan))

    # over/under implied
    if 'B365>2.5' in combined.columns:
        out['imp_over2_5'] = implied_prob(combined['B365>2.5'].values)
    if 'B365<2.5' in combined.columns:
        out['imp_under2_5'] = implied_prob(combined['B365<2.5'].values)

    # basic market summary features
    if 'MaxH' in combined.columns and 'AvgH' in combined.columns:
        out['spread_max_avg_H'] = combined['MaxH'] - combined['AvgH']
    if 'Bb1X2' in combined.columns:
        out['bookies_count'] = combined['Bb1X2']

    # parse Date properly and sort
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date'], dayfirst=True, errors='coerce')
    out = out.sort_values('Date', na_position='last')

    # save
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print('Wrote combined matches to', OUT)
    print('Rows:', len(out))
    print('Sample columns:', list(out.columns)[:40])


if __name__ == '__main__':
    main()
