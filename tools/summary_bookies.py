import pandas as pd
p = r'C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\simulation_evaluation.csv'

df = pd.read_csv(p)
total = len(df)
has_bookie = df['max_bookie_odds'].notna().sum() if 'max_bookie_odds' in df.columns else 0
better = (df['max_bookie_odds'] > df['fair_odds']).sum() if ('max_bookie_odds' in df.columns and 'fair_odds' in df.columns) else 0
acc = df['correct'].astype(bool).sum() / total * 100

print(f'Total rows: {total}')
print(f'Rows with max_bookie_odds: {has_bookie}')
print(f'Rows where max_bookie_odds > fair_odds: {better}')
print(f'Overall accuracy: {acc:.2f}%')

if better > 0:
    sel = df[(df['max_bookie_odds'] > df['fair_odds']) & (df['correct'] == True)]
    if len(sel) == 0:
        print('\nNo rows where bookmaker > fair and prediction was correct.')
    else:
        print('\nSample rows where bookmaker > fair and predicted was correct (first 5):')
        cols = ['Date','Time','HomeTeam','AwayTeam','pred_FTR','predicted','predicted_pct','fair_odds','max_bookie_odds','max_bookie']
        cols = [c for c in cols if c in sel.columns]
        print(sel[cols].head().to_string(index=False))

# Show top 5 matches with largest (max_bookie_odds - fair_odds)
if 'max_bookie_odds' in df.columns and 'fair_odds' in df.columns:
    df['bookie_gap'] = df['max_bookie_odds'] - df['fair_odds']
    top = df.sort_values('bookie_gap', ascending=False).head(5)
    print('\nTop 5 matches by bookmaker - fair odds gap:')
    cols2 = ['Date','HomeTeam','AwayTeam','predicted','predicted_pct','fair_odds','max_bookie_odds','max_bookie','bookie_gap']
    cols2 = [c for c in cols2 if c in top.columns]
    print(top[cols2].to_string(index=False))

# Save a small CSV of opportunities where bookmaker > fair
if better > 0:
    opp = df[df['max_bookie_odds'] > df['fair_odds']]
    out = r'C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\simulation_bookie_opportunities.csv'
    opp.to_csv(out, index=False)
    print(f"\nSaved {len(opp)} opportunity rows to {out}")
