import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / 'artifacts'
GRID = ART / 'betting_threshold_grid.csv'
GRID_UNW = ART / 'betting_threshold_grid_unweighted.csv'

def load_grid(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df

def rank_and_filter(df, min_bets=30, top_k=5, label='combined'):
    sub = df[df['label'] == label].copy()
    if sub.empty:
        return pd.DataFrame()
    # require minimum bets
    sub['ok'] = sub['n_bets'] >= min_bets
    cand = sub[sub['ok']].copy()
    if cand.empty:
        # relax if nothing
        cand = sub.copy()
    cand = cand.sort_values(['roi_pct', 'final_capital'], ascending=False)
    return cand.head(top_k)

def stability_across_seasons(df_all):
    # expects df_all to contain 'label' column with season names and 'multiplier' and 'min_prob'
    # pivot to compute mean and std of roi across labels per (multiplier,min_prob)
    if 'label' not in df_all.columns:
        return None
    pivot = df_all.pivot_table(index=['multiplier','min_prob'], columns='label', values='roi_pct')
    stats = pd.DataFrame({
        'mean_roi': pivot.mean(axis=1),
        'std_roi': pivot.std(axis=1),
        'n_seasons': pivot.count(axis=1)
    })
    stats = stats.reset_index()
    stats = stats.sort_values(['mean_roi','std_roi'], ascending=[False, True])
    return stats

def masked_heatmap(sub, metric='roi_pct', min_bets=30, out_name=None):
    pivot = sub.pivot(index='min_prob', columns='multiplier', values=metric)
    mask = sub.pivot(index='min_prob', columns='multiplier', values='n_bets') < min_bets
    plt.figure(figsize=(8,5))
    cmap = plt.get_cmap('RdYlGn')
    im = plt.imshow(pivot, aspect='auto', origin='lower', cmap=cmap)
    # overlay hatch for masked cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if mask.iloc[i,j]:
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, hatch='///', edgecolor='gray'))
    plt.colorbar(im, label=metric)
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(c) for c in pivot.index])
    plt.xlabel('Multiplier')
    plt.ylabel('Min prob')
    if out_name:
        plt.title(out_name)
        plt.tight_layout()
        outfile = ART / (out_name.replace(' ', '_') + '.png')
        plt.savefig(outfile, dpi=150)
        plt.close()

def main():
    grid = load_grid(GRID)
    grid_unw = load_grid(GRID_UNW)
    if grid is None and grid_unw is None:
        print('No grid files found')
        return

    # Use combined label if present
    label = 'combined'
    results = []
    if grid is not None:
        top = rank_and_filter(grid, min_bets=30, top_k=10, label=label)
        print('Top configs (weighted grid):')
        print(top[['multiplier','min_prob','n_bets','roi_pct','final_capital','total_staked','total_profit']])
        results.append(('weighted', top))
        # produce masked heatmap
        sub = grid[grid['label']==label]
        masked_heatmap(sub, 'roi_pct', min_bets=30, out_name='recommended_roi_weighted')

    if grid_unw is not None:
        topu = rank_and_filter(grid_unw, min_bets=30, top_k=10, label=label)
        print('\nTop configs (unweighted grid):')
        print(topu[['multiplier','min_prob','n_bets','roi_pct','final_capital','total_staked','total_profit']])
        results.append(('unweighted', topu))
        subu = grid_unw[grid_unw['label']==label]
        masked_heatmap(subu, 'roi_pct', min_bets=30, out_name='recommended_roi_unweighted')

    # stability across seasons if seasons present in original df (we try to detect)
    # attempt to read original simulation_evaluation.csv for season column
    sim = ART.parent / 'artifacts' / 'simulation_evaluation.csv'
    sim_file = ART.parent / 'artifacts' / 'simulation_evaluation.csv'
    # alternatively attempt to use season labels present in grid
    if grid is not None and 'label' in grid.columns and grid['label'].nunique() > 1:
        stats = stability_across_seasons(grid)
        if stats is not None:
            out = ART / 'thresholds_stability.csv'
            stats.to_csv(out, index=False)
            print(f'Saved stability stats to {out}')

    # consolidate recommended configs to CSV
    recs = pd.concat([t[1] for t in results if not t[1].empty], ignore_index=True) if results else pd.DataFrame()
    if not recs.empty:
        out_recs = ART / 'recommended_thresholds.csv'
        recs.to_csv(out_recs, index=False)
        print(f'Saved recommended thresholds to {out_recs}')

if __name__ == '__main__':
    main()
