"""Prepare and clean match CSVs from `partite/` into a single dataframe.
Saves cleaned CSV to `data/combined_matches.csv`.
"""
import pandas as pd
import glob
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
PARTITE_DIR = os.path.join(ROOT, 'partite')
OUT = os.path.join(ROOT, 'data', 'combined_matches.csv')

def canonicalize_team(name):
    if pd.isna(name):
        return name
    return name.strip()

def read_all_seasons():
    files = sorted(glob.glob(os.path.join(PARTITE_DIR, 'season-*.csv')))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            # try with semicolon
            df = pd.read_csv(f, sep=';')
        df['source_file'] = os.path.basename(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def clean(df):
    # parse date dd/mm/yy
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
    # canonicalize teams
    df['HomeTeam'] = df['HomeTeam'].apply(canonicalize_team)
    df['AwayTeam'] = df['AwayTeam'].apply(canonicalize_team)
    # numeric cols
    num_cols = ['FTHG','FTAG','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop duplicates
    df = df.drop_duplicates()
    # filter rows without date or teams
    df = df.dropna(subset=['Date','HomeTeam','AwayTeam'])
    # create result label
    if 'FTR' in df.columns:
        df['Result'] = df['FTR']
    else:
        # fallback
        df['Result'] = df.apply(lambda r: 'H' if r['FTHG']>r['FTAG'] else ('A' if r['FTAG']>r['FTHG'] else 'D'), axis=1)
    return df


def main():
    df = read_all_seasons()
    df = clean(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print('Saved', OUT)

if __name__ == '__main__':
    main()
