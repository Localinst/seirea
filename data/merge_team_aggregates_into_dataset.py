"""Merge team_aggregates.csv into the cleaned dataset with Pts.

Outputs: dataset_cleaned_for_ml_with_pts_teamagg.csv in project root.

Heuristics: exact match on season + squad (case-insensitive, stripped).
If unmatched rows remain, attempt a contains-based mapping (team name contains or vice-versa).
Prints a short report of matches/unmatched samples.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
TEAM_AGG = DATA_DIR / 'team_aggregates.csv'

# search for the dataset file in repo root
candidates = list(ROOT.glob('dataset_cleaned_for_ml_with*pts*.csv')) + list(ROOT.glob('dataset_cleaned_for_ml_with*pts*.CSV'))
if not candidates:
    # try alternative underscored name
    candidates = list(ROOT.glob('dataset_cleaned_for_ml_with_pts*.csv'))

if not candidates:
    raise SystemExit('Could not find dataset_cleaned_for_ml_with_pts CSV in repo root. Expected patterns dataset_cleaned_for_ml_with*pts*.csv or dataset_cleaned_for_ml_with_pts*.csv')

dataset_fp = candidates[0]
print('Using dataset:', dataset_fp)

# Read files
df = pd.read_csv(dataset_fp, encoding='utf-8')
tag = pd.read_csv(TEAM_AGG, encoding='utf-8')

# normalize keys
def norm(s):
    if pd.isna(s):
        return ''
    return str(s).strip().lower()

for df_ in (df, tag):
    df_['__season_norm'] = df_['season'].apply(norm) if 'season' in df_.columns else df_.get('Season', '').apply(norm)
    if 'squad' not in df_.columns and 'team' in df_.columns:
        df_['squad'] = df_['team']
    df_['__squad_norm'] = df_['squad'].apply(norm) if 'squad' in df_.columns else ''

if 'season' not in tag.columns:
    raise SystemExit('team_aggregates.csv missing "season" column')
if 'squad' not in tag.columns:
    raise SystemExit('team_aggregates.csv missing "squad" column')

tag['__season_norm'] = tag['season'].apply(norm)
tag['__squad_norm'] = tag['squad'].apply(norm)

# first, exact merge
merged = df.merge(tag.drop(columns=['season','squad']), how='left', left_on=['__season_norm','__squad_norm'], right_on=['__season_norm','__squad_norm'], suffixes=('','_team'))

# report unmatched
unmatched = merged[merged['goals_tot'].isna()]
print(f'Total rows in dataset: {len(df)}')
print(f'Matched rows after exact join: {len(df) - len(unmatched)}')

# attempt fuzzy contains-based mapping for unmatched
if not unmatched.empty:
    tag_map = {(r['__season_norm'], r['__squad_norm']): r for _, r in tag.iterrows()}
    fill_count = 0
    for idx, row in unmatched.iterrows():
        s = row['__season_norm']
        q = row['__squad_norm']
        if not q:
            continue
        # try to find tag squad containing q or q containing tag squad for same season
        candidates = [r for _, r in tag[tag['__season_norm'] == s].iterrows()]
        found = None
        for r in candidates:
            t = r['__squad_norm']
            if t == q:
                found = r
                break
            if q in t or t in q:
                found = r
                break
        if found is not None:
            # copy numeric columns from found to merged at idx
            cols = ['roster_size','starters_count','age_mean','age_mean_starters','pct_under23','pct_gk','pct_df','pct_mf','pct_fw','goals_tot','assists_tot','xg_tot','xga_proxy','team_xg_per90','conversion_sh_to_g','progressions_tot','yellow_cards','red_cards','top5_mean_gpa','bottom5_mean_gpa','gap_top5_bottom5_gpa','age_std']
            for c in cols:
                if c in found.index and c in merged.columns:
                    merged.at[idx, c] = found[c]
                elif c in found.index and c not in merged.columns:
                    merged.at[idx, c] = found[c]
            fill_count += 1
    print(f'Filled {fill_count} rows using contains-based heuristic')

# final unmatched
final_unmatched = merged[merged['goals_tot'].isna()]
print(f'Final unmatched rows: {len(final_unmatched)}')
if len(final_unmatched) > 0:
    print('Sample unmatched (season, squad):')
    print(final_unmatched[['season','squad']].drop_duplicates().head(20).to_string(index=False))

# write merged output
out_fp = ROOT / 'dataset_cleaned_for_ml_with_pts_teamagg.csv'
merged.to_csv(out_fp, index=False, encoding='utf-8-sig')
print('WROTE', out_fp)
