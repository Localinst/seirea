"""Normalize bookmaker odds into implied probabilities and add simple market features.

Reads data/combined_matches.csv, computes implied probabilities from AvgH/AvgD/AvgA (falls back to B365* if Avg* missing), removes the overround (vig) by normalization, and writes back to data/combined_matches.csv (overwrites) and also saves a backup data/combined_matches_markets.csv.

This script is safe to re-run.
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
INPATH = os.path.join(ROOT, 'data', 'combined_matches.csv')
BACKUP = os.path.join(ROOT, 'data', 'combined_matches_markets.csv')


def safe_inv(x):
    try:
        if pd.isna(x) or float(x) == 0:
            return np.nan
        return 1.0 / float(x)
    except Exception:
        return np.nan


def compute_implied_from_cols(df, prefix):
    # prefix expected like 'Avg' to find AvgH, AvgD, AvgA
    hcol = f'{prefix}H'
    dcol = f'{prefix}D'
    acol = f'{prefix}A'
    if not (hcol in df.columns and dcol in df.columns and acol in df.columns):
        return None
    inv_h = df[hcol].apply(safe_inv)
    inv_d = df[dcol].apply(safe_inv)
    inv_a = df[acol].apply(safe_inv)
    s = inv_h + inv_d + inv_a
    # avoid divide by zero
    imp_h = inv_h / s
    imp_d = inv_d / s
    imp_a = inv_a / s
    return imp_h, imp_d, imp_a, s


def main():
    if not os.path.exists(INPATH):
        print('No combined_matches.csv found at', INPATH)
        return
    df = pd.read_csv(INPATH, low_memory=False)
    # try to parse Date if not parsed
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        except Exception:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # compute from Avg columns first
    avg_res = compute_implied_from_cols(df, 'Avg')
    b365_res = compute_implied_from_cols(df, 'B365')

    if avg_res is not None:
        imp_h, imp_d, imp_a, raw_sum = avg_res
        df['imp_H'] = imp_h
        df['imp_D'] = imp_d
        df['imp_A'] = imp_a
        df['market_raw_sum'] = raw_sum
        df['market_vig'] = raw_sum - 1.0
    elif b365_res is not None:
        imp_h, imp_d, imp_a, raw_sum = b365_res
        df['imp_H'] = imp_h
        df['imp_D'] = imp_d
        df['imp_A'] = imp_a
        df['market_raw_sum'] = raw_sum
        df['market_vig'] = raw_sum - 1.0
    else:
        print('No Avg or B365 odds columns found to compute implied probabilities.')

    # If both exist compute both and add comparison
    if (avg_res is not None) and (b365_res is not None):
        bimp_h, bimp_d, bimp_a, braw_sum = b365_res
        df['imp_B365_H'] = bimp_h
        df['imp_B365_D'] = bimp_d
        df['imp_B365_A'] = bimp_a
        df['imp_imp_diff_H'] = df['imp_H'] - df['imp_B365_H']
        df['imp_imp_diff_D'] = df['imp_D'] - df['imp_B365_D']
        df['imp_imp_diff_A'] = df['imp_A'] - df['imp_B365_A']

    # simple spread/volatility features using Max vs Avg if present
    for col in ['H', 'D', 'A']:
        maxc = f'Max{col}'
        avgc = f'Avg{col}'
        if maxc in df.columns and avgc in df.columns:
            df[f'odds_spread_{col}'] = df[maxc] - df[avgc]
        else:
            df[f'odds_spread_{col}'] = np.nan

    # basic bookie counts if present in Bb1X2 or similar
    if 'Bb1X2' in df.columns:
        df['has_bookies_1x2'] = ~df['Bb1X2'].isna()
    
    # save backup and overwrite INPATH
    os.makedirs(os.path.dirname(INPATH), exist_ok=True)
    df.to_csv(BACKUP, index=False)
    df.to_csv(INPATH, index=False)
    print('Wrote enriched combined_matches to', INPATH)
    print('Backup saved to', BACKUP)
    print('Rows:', len(df))
    print('Sample cols:', df.columns.tolist()[:80])


if __name__ == '__main__':
    main()
