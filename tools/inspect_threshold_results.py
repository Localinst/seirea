import pandas as pd
from pathlib import Path

p = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\betting_threshold_grid.csv")
if not p.exists():
    raise SystemExit(f"Missing {p}")

df = pd.read_csv(p)
if 'label' not in df.columns:
    df['label'] = 'combined'

mult = 1.25
print(f"Inspecting multiplier={mult} per label\n")
for lab in sorted(df['label'].unique()):
    sub = df[df['label'] == lab]
    sub_m = sub[sub['multiplier'] == mult]
    if sub_m.empty:
        print(f"Label {lab}: no rows for multiplier {mult}")
        continue
    print(f"Label {lab} - multiplier {mult} rows:")
    print(sub_m[['min_prob','n_bets','wins','winrate_pct','final_capital','roi_pct']].to_string(index=False))
    best = sub.sort_values('roi_pct', ascending=False).head(1)
    print("Best overall for label by ROI:")
    print(best[['multiplier','min_prob','n_bets','wins','final_capital','roi_pct']].to_string(index=False))
    print()

print("Overall top 5 by ROI across labels:")
print(df.sort_values('roi_pct', ascending=False).head(5)[['label','multiplier','min_prob','n_bets','roi_pct']].to_string(index=False))
