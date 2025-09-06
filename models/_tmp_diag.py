import pandas as pd, os
ROOT = os.getcwd()
FEATS = os.path.join(ROOT,'data','features.csv')
COM = os.path.join(ROOT,'data','combined_matches.csv')
feats = pd.read_csv(FEATS, nrows=5)
comb = pd.read_csv(COM, nrows=5)
print('features columns:', list(feats.columns))
print('combined columns:', list(comb.columns))
cols = ['Date','HomeTeam','AwayTeam']
for c in ('Result','FTR','FTHG','FTAG'):
    if c in comb.columns and c not in cols:
        cols.append(c)
comb_slim = comb[cols].copy()
print('comb_slim columns:', list(comb_slim.columns))
print('Result in comb_slim?', 'Result' in comb_slim.columns)
if 'FTR' in comb_slim.columns:
    print('Sample FTR:', comb_slim['FTR'].head().tolist())
if 'FTHG' in comb_slim.columns and 'FTAG' in comb_slim.columns:
    print('Sample scores:', list(zip(comb_slim['FTHG'].head().tolist(), comb_slim['FTAG'].head().tolist())))
