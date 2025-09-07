import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / 'artifacts'
GRID = ART / 'betting_threshold_grid.csv'
GRID_UNW = ART / 'betting_threshold_grid_unweighted.csv'

def pivot_and_plot(df, label_col='label', out_prefix='betting_threshold'):
    metrics = ['roi_pct', 'n_bets', 'avg_profit_per_bet', 'total_staked']
    for label in df[label_col].unique():
        sub = df[df[label_col] == label]
        if sub.empty:
            continue
        # assess sparsity: count cells with n_bets <= threshold
        low_count = (sub['n_bets'] <= 5).sum()
        total_cells = len(sub)
        print(f"Label={label}: {total_cells} cells, low-count (<=5 bets) cells={low_count}")

        # for each metric create heatmap
        for metric in metrics:
            if metric not in sub.columns:
                continue
            try:
                pivot = sub.pivot(index='min_prob', columns='multiplier', values=metric)
            except Exception:
                continue
            if pivot.empty:
                continue
            plt.figure(figsize=(8,5))
            im = plt.imshow(pivot, aspect='auto', cmap='RdYlGn', origin='lower')
            plt.colorbar(im, label=metric)
            plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(c) for c in pivot.index])
            plt.xlabel('Multiplier (fair * m)')
            plt.ylabel('Min predicted probability')
            plt.title(f"{metric} heatmap ({label})")
            # annotate values
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.iloc[i,j]
                    txt = f"{v:.2f}" if pd.notna(v) else 'NA'
                    plt.text(j, i, txt, ha='center', va='center', color='black', fontsize=8)
            out_file = ART / f"{out_prefix}_{metric}_{label}.png"
            plt.tight_layout()
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"Saved heatmap: {out_file}")

def analyze_and_plot():
    if GRID.exists():
        print(f"Reading {GRID}")
        df = pd.read_csv(GRID)
        pivot_and_plot(df, label_col='label', out_prefix='betting_threshold_grid')
    else:
        print(f"Grid file not found: {GRID}")

    if GRID_UNW.exists():
        print(f"Reading {GRID_UNW}")
        df_unw = pd.read_csv(GRID_UNW)
        pivot_and_plot(df_unw, label_col='label', out_prefix='betting_threshold_grid_unweighted')
    else:
        print(f"Unweighted grid file not found: {GRID_UNW}")

if __name__ == '__main__':
    analyze_and_plot()
