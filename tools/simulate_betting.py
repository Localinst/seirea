import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# base paths
root = Path(__file__).resolve().parents[1]
art = root / 'artifacts'
p_in = art / 'simulation_evaluation.csv'

df = pd.read_csv(p_in, parse_dates=["Date"], dayfirst=True)
# Ensure chronological order
if "Time_sort" in df.columns:
    df = df.sort_values(["Date","Time_sort"]) if "Time_sort" in df.columns else df.sort_values("Date")
else:
    df = df.sort_values("Date")

# Parameters
starting_capital = 100.0
stake = 1.0
threshold = 1.25  # 10% higher than fair odds
min_fair = 1.5   # only bet if fair odds >= this (avoid very low odds bets)

# Prepare columns
df = df.reset_index(drop=True)
df['match_index'] = df.index + 1

def run_betting_sim(df_in: pd.DataFrame, out_csv: Path, out_plot: Path, label: str):
    df = df_in.copy()
    # Ensure chronological order
    if 'DateTime_sort' in df.columns:
        df = df.sort_values(['DateTime_sort', 'Date'])
    elif 'Time' in df.columns:
        try:
            df['DateTime_sort'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), dayfirst=True, errors='coerce')
            df = df.sort_values(['DateTime_sort'])
        except Exception:
            df = df.sort_values('Date')
    else:
        df = df.sort_values('Date')

    # Parameters
    starting_capital = 100.0
    stake = 1.0
    threshold = 1.25
    min_fair = 1.4

    df = df.reset_index(drop=True)
    df['match_index'] = df.index + 1

    bets = []
    capital = starting_capital
    capital_history = []
    accuracy_history = []

    cum_correct = 0
    total_seen = 0

    for i, row in df.iterrows():
        total_seen += 1
        if bool(row.get('correct')):
            cum_correct += 1
        accuracy_history.append(cum_correct / total_seen if total_seen>0 else 0)
        capital_history.append(capital)

        fair = row.get('fair_odds')
        max_bookie = row.get('max_bookie_odds')
        if pd.isna(fair) or pd.isna(max_bookie):
            continue
        try:
            fair = float(fair)
            max_bookie = float(max_bookie)
        except Exception:
            continue

        if max_bookie >= fair * threshold and fair >= min_fair:
            pred_correct = bool(row.get('correct'))
            if pred_correct:
                profit = stake * (max_bookie - 1.0)
                capital += profit
            else:
                profit = -stake
                capital += profit
            bets.append({
                'match_index': int(row['match_index']),
                'Date': row.get('Date'),
                'HomeTeam': row.get('HomeTeam'),
                'AwayTeam': row.get('AwayTeam'),
                'predicted': row.get('predicted'),
                'predicted_pct': row.get('predicted_pct'),
                'fair_odds': fair,
                'max_bookie_odds': max_bookie,
                'max_bookie': row.get('max_bookie'),
                'stake': stake,
                'profit': profit,
                'capital_after': round(capital,2),
                'pred_correct': pred_correct
            })

    capital_history.append(capital)
    accuracy_history.append(cum_correct / total_seen if total_seen>0 else 0)

    bets_df = pd.DataFrame(bets)
    bets_df.to_csv(out_csv, index=False)

    n_bets = len(bets_df)
    wins = int(bets_df['pred_correct'].sum()) if n_bets>0 else 0
    losses = n_bets - wins
    final_capital = capital
    roi = (final_capital - starting_capital) / starting_capital * 100

    print(f"[{label}] Bets placed: {n_bets}")
    print(f"[{label}] Wins: {wins}, Losses: {losses}")
    print(f"[{label}] Final capital: {final_capital:.2f} EUR (ROI {roi:.2f}%)")

    # plotting
    try:
        plt.style.use('seaborn-darkgrid')
    except Exception:
        try:
            plt.style.use('seaborn')
        except Exception:
            plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(range(1, len(capital_history)+1), capital_history, label='Capital (EUR)', color='tab:green')
    ax1.set_xlabel('Match step')
    ax1.set_ylabel('Capital (EUR)', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(accuracy_history)+1), np.array(accuracy_history)*100, label='Cumulative accuracy (%)', color='tab:blue', alpha=0.8)
    ax2.set_ylabel('Cumulative accuracy (%)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    if n_bets>0:
        bet_indices = bets_df['match_index'].tolist()
        bet_caps = bets_df['capital_after'].tolist()
        ax1.scatter(bet_indices, bet_caps, color='red', s=20, label='Bet events')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='upper left')
    ax1.set_title(f"Betting simulation ({label}): stake=€{stake}, threshold={int((threshold-1)*100)}% above fair odds")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=200)
    print(f"Saved betting CSV to {out_csv}")
    print(f"Saved betting plot to {out_plot}")


# Run per-sim_source (e.g., simulation_2023_2024_predictions.csv) and combined
if 'sim_source' in df.columns:
    sources = df['sim_source'].unique().tolist()
else:
    sources = [None]

combined_out_csv = art / 'simulation_betting_results_combined.csv'
combined_out_plot = art / 'simulation_evaluation_betting_combined.png'
run_betting_sim(df, combined_out_csv, combined_out_plot, 'combined')

for src in sources:
    if src is None:
        continue
    subset = df[df['sim_source'] == src].copy()
    # create safe name
    safe = src.replace('.csv','')
    out_csv = art / f'simulation_betting_results_{safe}.csv'
    out_plot = art / f'simulation_evaluation_betting_{safe}.png'
    run_betting_sim(subset, out_csv, out_plot, safe)
