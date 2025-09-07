import pandas as pd
from pathlib import Path

art = Path(__file__).resolve().parents[1] / 'artifacts'
w_csv = art / 'betting_threshold_grid.csv'
uw_csv = art / 'betting_threshold_grid_unweighted.csv'
out_csv = art / 'best_filters_summary.csv'

frames = []
if w_csv.exists():
    dfw = pd.read_csv(w_csv)
    dfw['mode'] = 'weighted'
    # weighted doesn't have min_fair
    frames.append(dfw)
if uw_csv.exists():
    dfuw = pd.read_csv(uw_csv)
    dfuw['mode'] = 'unweighted'
    frames.append(dfuw)

if not frames:
    raise SystemExit('No grid CSVs found in artifacts')

df = pd.concat(frames, ignore_index=True, sort=False)

# Normalize columns
if 'min_fair' not in df.columns:
    df['min_fair'] = pd.NA

# Ensure numeric types
for c in ['multiplier','min_prob','n_bets','wins','final_capital','roi_pct','min_fair']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Save top results overall and per label
records = []

# Overall top 20 by roi
top_overall = df.sort_values('roi_pct', ascending=False).head(20)
for _, r in top_overall.iterrows():
    records.append({
        'scope': 'overall',
        'mode': r.get('mode'),
        'label': r.get('label'),
        'min_fair': r.get('min_fair'),
        'multiplier': r.get('multiplier'),
        'min_prob': r.get('min_prob'),
        'n_bets': r.get('n_bets'),
        'wins': r.get('wins'),
        'final_capital': r.get('final_capital'),
        'roi_pct': r.get('roi_pct')
    })

# Best per label
for lab in sorted(df['label'].dropna().unique()):
    sub = df[df['label'] == lab]
    if sub.empty:
        continue
    best = sub.sort_values('roi_pct', ascending=False).head(5)
    for _, r in best.iterrows():
        records.append({
            'scope': f'label:{lab}',
            'mode': r.get('mode'),
            'label': r.get('label'),
            'min_fair': r.get('min_fair'),
            'multiplier': r.get('multiplier'),
            'min_prob': r.get('min_prob'),
            'n_bets': r.get('n_bets'),
            'wins': r.get('wins'),
            'final_capital': r.get('final_capital'),
            'roi_pct': r.get('roi_pct')
        })

# Best with n_bets >= 30 (to avoid small-sample spikes)
sub_large = df[df['n_bets'] >= 30]
top_large = sub_large.sort_values('roi_pct', ascending=False).head(20)
for _, r in top_large.iterrows():
    records.append({
        'scope': 'overall_n_bets>=30',
        'mode': r.get('mode'),
        'label': r.get('label'),
        'min_fair': r.get('min_fair'),
        'multiplier': r.get('multiplier'),
        'min_prob': r.get('min_prob'),
        'n_bets': r.get('n_bets'),
        'wins': r.get('wins'),
        'final_capital': r.get('final_capital'),
        'roi_pct': r.get('roi_pct')
    })

out_df = pd.DataFrame.from_records(records)
out_df.to_csv(out_csv, index=False)
print(f"Wrote summary to {out_csv}")
print('\nTop 10 overall by ROI:')
print(top_overall[['mode','label','multiplier','min_prob','min_fair','n_bets','wins','final_capital','roi_pct']].head(10).to_string(index=False))
