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

starting_capital = 0
stake = 1.0

# (results will be built by run_grid; drop unused 'results' placeholder)

# Helper: effective multiplier weighted by probability.
# We reduce the required multiplier as predicted probability increases so
# high-confidence predictions need a smaller premium. Formula used:
# m_eff = 1 + (m - 1) * (1 - prob)
def effective_multiplier(base_m: float, prob: float) -> float:
    try:
        prob = float(prob)
    except Exception:
        return base_m
    prob = max(0.0, min(1.0, prob))
    return 1.0 + (base_m - 1.0) * (1.0 - prob)


# If the dataframe contains a season column, run per-season and combined.
season_col = None
for c in ['Season', 'season', 'SeasonName']:
    if c in df.columns:
        season_col = c
        break

def run_grid(on_df: pd.DataFrame, label: str) -> pd.DataFrame:
    res = []
    for m in multipliers:
        for pmin in min_probs:
            capital = starting_capital
            n_bets = 0
            wins = 0
            profits = []
            total_staked = 0.0
            total_profit = 0.0
            for _, row in on_df.iterrows():
                try:
                    fair = float(row.get('fair_odds'))
                    maxb = float(row.get('max_bookie_odds'))
                    prob = float(row.get('predicted_pct'))
                except Exception:
                    continue
                if pd.isna(fair) or pd.isna(maxb) or pd.isna(prob):
                    continue
                if prob < pmin:
                    continue
                m_eff = effective_multiplier(m, prob)
                # require bookie >= fair * m_eff
                if maxb >= fair * m_eff:
                    n_bets += 1
                    total_staked += stake
                    pred_correct = bool(row.get('correct'))
                    if pred_correct:
                        profit = stake * (maxb - 1.0)
                        capital += profit
                        total_profit += profit
                        wins += 1
                        profits.append(profit)
                    else:
                        profit = -stake
                        capital += profit
                        total_profit += profit
                        profits.append(profit)
            final_capital = capital
            # ROI on staked (match simulate_betting logic). If no stake, ROI = 0
            if total_staked > 0:
                roi = (total_profit / total_staked) * 100
            else:
                roi = 0.0
            winrate = (wins / n_bets * 100) if n_bets>0 else 0
            avg_profit = np.mean(profits) if len(profits)>0 else 0
            expected_value = np.mean(profits) if len(profits)>0 else 0
            res.append({
                'label': label,
                'multiplier': m,
                'min_prob': pmin,
                'n_bets': n_bets,
                'wins': wins,
                'winrate_pct': winrate,
                'final_capital': final_capital,
                'roi_pct': roi,
                'total_staked': total_staked,
                'total_profit': total_profit,
                'avg_profit_per_bet': avg_profit,
                'ev_per_bet': expected_value
            })
    return pd.DataFrame(res)


if season_col is not None:
    all_res = []
    for season in sorted(df[season_col].dropna().unique()):
        sub = df[df[season_col] == season]
        print(f"Running grid for season {season}, {len(sub)} rows")
        all_res.append(run_grid(sub, str(season)))
    # combined
    all_res.append(run_grid(df, 'combined'))
    res_df = pd.concat(all_res, ignore_index=True)
else:
    res_df = run_grid(df, 'combined')

# Save combined CSV
res_df.to_csv(out_csv, index=False)
print(f"Saved grid results to {out_csv}")

# For each label (season or 'combined') generate a heatmap and print bests
for label in res_df['label'].unique():
    sub = res_df[res_df['label'] == label]
    if sub.empty:
        continue
    best_roi = sub.sort_values('roi_pct', ascending=False).iloc[0]
    best_cap = sub.sort_values('final_capital', ascending=False).iloc[0]
    print(f"Label {label} Best by ROI:", best_roi.to_dict())
    print(f"Label {label} Best by final capital:", best_cap.to_dict())

    # Pivot for heatmap (ROI)
    pivot = sub.pivot(index='min_prob', columns='multiplier', values='roi_pct')
    plt.figure(figsize=(10,6))
    plt.imshow(pivot, aspect='auto', cmap='RdYlGn', origin='lower')
    plt.colorbar(label='ROI %')
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(c) for c in pivot.index])
    plt.xlabel('Multiplier (fair * m)')
    plt.ylabel('Min predicted probability')
    plt.title(f'ROI % heatmap by multiplier and min_prob ({label})')
    # annotate
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i,j]
            txt = f"{v:.1f}" if not pd.isna(v) else 'NA'
            plt.text(j, i, txt, ha='center', va='center', color='black')
    plt.tight_layout()
    out_plot_label = out_plot.parent / f"betting_threshold_heatmap_{label}.png"
    plt.savefig(out_plot_label, dpi=200)
    print(f"Saved heatmap to {out_plot_label}")


# --- Now run an 'unweighted' exploration that mirrors simulate_betting.py filters ---
print('\nRunning unweighted exploration to match simulate_betting filters (multiplier + min_fair + min_prob)')
min_fairs = [1.4, 1.5, 1.6]
def run_grid_unweighted(on_df: pd.DataFrame, label: str, min_fair_val: float) -> pd.DataFrame:
    res = []
    for m in multipliers:
        for pmin in min_probs:
            capital = starting_capital
            n_bets = 0
            wins = 0
            profits = []
            total_staked = 0.0
            total_profit = 0.0
            for _, row in on_df.iterrows():
                try:
                    fair = float(row.get('fair_odds'))
                    maxb = float(row.get('max_bookie_odds'))
                    prob = float(row.get('predicted_pct'))
                except Exception:
                    continue
                if pd.isna(fair) or pd.isna(maxb) or pd.isna(prob):
                    continue
                if prob < pmin:
                    continue
                if fair < min_fair_val:
                    continue
                # unweighted: require bookie >= fair * m
                if maxb >= fair * m:
                    n_bets += 1
                    total_staked += stake
                    pred_correct = bool(row.get('correct'))
                    if pred_correct:
                        profit = stake * (maxb - 1.0)
                        capital += profit
                        total_profit += profit
                        wins += 1
                        profits.append(profit)
                    else:
                        profit = -stake
                        capital += profit
                        total_profit += profit
                        profits.append(profit)
            final_capital = capital
            if total_staked > 0:
                roi = (total_profit / total_staked) * 100
            else:
                roi = 0.0
            winrate = (wins / n_bets * 100) if n_bets>0 else 0
            avg_profit = np.mean(profits) if len(profits)>0 else 0
            expected_value = np.mean(profits) if len(profits)>0 else 0
            res.append({
                'label': label,
                'min_fair': min_fair_val,
                'multiplier': m,
                'min_prob': pmin,
                'n_bets': n_bets,
                'wins': wins,
                'winrate_pct': winrate,
                'final_capital': final_capital,
                'roi_pct': roi,
                'total_staked': total_staked,
                'total_profit': total_profit,
                'avg_profit_per_bet': avg_profit,
                'ev_per_bet': expected_value
            })
    return pd.DataFrame(res)

all_unweighted = []
if season_col is not None:
    labels = sorted(df[season_col].dropna().unique()) + ['combined']
else:
    labels = ['combined']
for label in labels:
    sub = df if label == 'combined' else df[df[season_col] == label]
    for mf in min_fairs:
        print(f"Running unweighted grid for label={label}, min_fair={mf}")
        all_unweighted.append(run_grid_unweighted(sub, str(label), mf))
unw_df = pd.concat(all_unweighted, ignore_index=True)
out_unw_csv = out_csv.parent / 'betting_threshold_grid_unweighted.csv'
unw_df.to_csv(out_unw_csv, index=False)
print(f"Saved unweighted grid results to {out_unw_csv}")

# Generate heatmaps per min_fair and label for unweighted results
for mf in sorted(unw_df['min_fair'].unique()):
    df_mf = unw_df[unw_df['min_fair'] == mf]
    for label in df_mf['label'].unique():
        sub = df_mf[df_mf['label'] == label]
        if sub.empty:
            continue
        pivot = sub.pivot(index='min_prob', columns='multiplier', values='roi_pct')
        plt.figure(figsize=(10,6))
        plt.imshow(pivot, aspect='auto', cmap='RdYlGn', origin='lower')
        plt.colorbar(label='ROI %')
        plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(c) for c in pivot.index])
        plt.xlabel('Multiplier (fair * m)')
        plt.ylabel('Min predicted probability')
        plt.title(f'Unweighted ROI % heatmap (min_fair={mf}) ({label})')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iloc[i,j]
                txt = f"{v:.1f}" if not pd.isna(v) else 'NA'
                plt.text(j, i, txt, ha='center', va='center', color='black')
        plt.tight_layout()
        out_plot_label = out_plot.parent / f"betting_threshold_unweighted_heatmap_minfair_{mf}_{label}.png"
        plt.savefig(out_plot_label, dpi=200)
        print(f"Saved unweighted heatmap to {out_plot_label}")
