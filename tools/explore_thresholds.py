import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

p_in = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\simulation_evaluation.csv")
out_csv = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\betting_threshold_grid.csv")
out_plot = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\betting_threshold_heatmap.png")

# Load
df = pd.read_csv(p_in, parse_dates=["Date"], dayfirst=True)
df = df.sort_values(["Date"]) if "Date" in df.columns else df

# Ensure numeric columns available
if 'predicted_pct' not in df.columns or 'max_bookie_odds' not in df.columns:
    raise SystemExit('Input CSV missing predicted_pct or max_bookie_odds columns')

# Grid parameters
multipliers = [1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5]
min_probs = [0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]

starting_capital = 100.0
stake = 1.0

results = []

for m in multipliers:
    for pmin in min_probs:
        capital = starting_capital
        n_bets = 0
        wins = 0
        profits = []
        for _, row in df.iterrows():
            try:
                fair = float(row.get('fair_odds'))
                maxb = float(row.get('max_bookie_odds'))
                prob = float(row.get('predicted_pct'))
            except Exception:
                continue
            if pd.isna(fair) or pd.isna(maxb) or pd.isna(prob):
                continue
            # require predicted probability >= pmin and bookie >= fair*m
            if prob >= pmin and maxb >= fair * m:
                n_bets += 1
                pred_correct = bool(row.get('correct'))
                if pred_correct:
                    profit = stake * (maxb - 1.0)
                    capital += profit
                    wins += 1
                    profits.append(profit)
                else:
                    profit = -stake
                    capital += profit
                    profits.append(profit)
        final_capital = capital
        roi = (final_capital - starting_capital) / starting_capital * 100
        winrate = (wins / n_bets * 100) if n_bets>0 else 0
        avg_profit = np.mean(profits) if len(profits)>0 else 0
        expected_value = np.mean(profits) if len(profits)>0 else 0
        results.append({
            'multiplier': m,
            'min_prob': pmin,
            'n_bets': n_bets,
            'wins': wins,
            'winrate_pct': winrate,
            'final_capital': final_capital,
            'roi_pct': roi,
            'avg_profit_per_bet': avg_profit,
            'ev_per_bet': expected_value
        })

res_df = pd.DataFrame(results)
res_df.to_csv(out_csv, index=False)
print(f"Saved grid results to {out_csv}")

# Find best by roi and by final_capital
best_roi = res_df.sort_values('roi_pct', ascending=False).iloc[0]
best_cap = res_df.sort_values('final_capital', ascending=False).iloc[0]
print('Best by ROI:', best_roi.to_dict())
print('Best by final capital:', best_cap.to_dict())

# Pivot for heatmap (ROI)
pivot = res_df.pivot(index='min_prob', columns='multiplier', values='roi_pct')
plt.figure(figsize=(10,6))
plt.imshow(pivot, aspect='auto', cmap='RdYlGn', origin='lower')
plt.colorbar(label='ROI %')
plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
plt.yticks(range(len(pivot.index)), [str(c) for c in pivot.index])
plt.xlabel('Multiplier (fair * m)')
plt.ylabel('Min predicted probability')
plt.title('ROI % heatmap by multiplier and min_prob')
# annotate
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.iloc[i,j]
        txt = f"{v:.1f}" if not pd.isna(v) else 'NA'
        plt.text(j, i, txt, ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig(out_plot, dpi=200)
print(f"Saved heatmap to {out_plot}")
