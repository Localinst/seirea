import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# base paths
root = Path(__file__).resolve().parents[1]
art = root / 'artifacts'
p_in = art / 'simulation_evaluation.csv'

# Load evaluation
df = pd.read_csv(p_in, parse_dates=["Date"], dayfirst=True)
# Ensure chronological order
if "Time_sort" in df.columns:
    df = df.sort_values(["Date","Time_sort"]) if "Time_sort" in df.columns else df.sort_values("Date")
else:
    df = df.sort_values("Date")

# Default parameters

stake = 1.0
# Defaults that may be overridden by best_filters_summary.csv
threshold = 1.25  # multiplier (bookie >= fair * threshold)
min_fair = 0  # only bet if fair odds >= this (avoid very low odds bets)
min_prob_filter = None
# If the max bookie isn't >= fair * threshold, still allow the bet when
# the absolute difference (max_bookie - fair) is small but non-negative.
# This accepts near-fair edges (e.g., max_bookie = fair + 0.15) as playable.
diff_threshold = 0
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

    df = df.reset_index(drop=True)
    df['match_index'] = df.index + 1

    bets = []
    capital = 0.0
    capital_history = []         # cumulative profit over time
    accuracy_history = []
    staked_history = []          # cumulative money staked over time
    profit_history = []          # same as capital_history (kept for clarity)
    bets_count_history = []      # cumulative number of bets placed

    cum_correct = 0
    total_seen = 0
    total_staked = 0.0           # tracked incrementally when a bet is placed
    total_profit = 0.0           # tracked incrementally (sum of profit outcomes)
    bets_placed_count = 0
    candidate_count = 0
    candidate_skipped = 0

    for i, row in df.iterrows():
        total_seen += 1
        # small debug: print first few loop steps
        if total_seen <= 12:
            try:
                print(f"LOOP DEBUG step={total_seen} idx={i} match_index={row.get('match_index')} total_staked={total_staked} capital={capital}")
            except Exception:
                pass
        if bool(row.get('correct')):
            cum_correct += 1

        fair = row.get('fair_odds')
        max_bookie = row.get('max_bookie_odds')
        if pd.isna(fair) or pd.isna(max_bookie):
            capital_history.append(capital)
            accuracy_history.append(cum_correct / total_seen if total_seen>0 else 0)
            staked_history.append(total_staked)
            bets_count_history.append(bets_placed_count)
            profit_history.append(total_profit)
            continue
        try:
            fair = float(fair)
            max_bookie = float(max_bookie)
        except Exception:
            capital_history.append(capital)
            accuracy_history.append(cum_correct / total_seen if total_seen>0 else 0)
            staked_history.append(total_staked)
            profit_history.append(total_profit)
            continue

        # apply min_prob filter if configured
        prob = row.get('predicted_pct')
        try:
            prob = float(prob)
        except Exception:
            prob = None

        # apply min_prob filter if configured
        if min_prob_filter is not None and (prob is None or prob < min_prob_filter):
            capital_history.append(capital)
            accuracy_history.append(cum_correct / total_seen if total_seen>0 else 0)
            staked_history.append(total_staked)
            bets_count_history.append(bets_placed_count)
            profit_history.append(total_profit)
            continue

        # evaluate betting condition
        # allow bet if either a) max_bookie meets multiplier threshold, or
        # b) max_bookie is at least fair and the absolute uplift over fair is
        #    small (<= diff_threshold) — this accepts near-fair offers
        cond_multiplier = (max_bookie >= fair * threshold)
        cond_small_uplift = (max_bookie >= fair and (max_bookie - fair) <= diff_threshold)
        cond = (fair >= min_fair) and (cond_multiplier or cond_small_uplift)
        candidate_count += 1
        if not cond:
            candidate_skipped += 1
            # record histories for this non-bet iteration
            capital_history.append(capital)
            accuracy_history.append(cum_correct / total_seen if total_seen>0 else 0)
            staked_history.append(total_staked)
            bets_count_history.append(bets_placed_count)
            profit_history.append(total_profit)
            continue
        else:
                # place bet
                bets_placed_count += 1
                total_staked += stake
                pred_correct = bool(row.get('correct'))
                if pred_correct:
                    profit = stake * (max_bookie - 1.0)
                    capital += profit
                else:
                    profit = -stake
                    capital += profit
                total_profit += profit
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
                # record histories for this bet iteration
                capital_history.append(capital)
                accuracy_history.append(cum_correct / total_seen if total_seen>0 else 0)
                staked_history.append(total_staked)
                bets_count_history.append(bets_placed_count)
                profit_history.append(total_profit)

    # histories are recorded per-iteration inside the loop

    bets_df = pd.DataFrame(bets)
    bets_df.to_csv(out_csv, index=False)

    n_bets = len(bets_df)
    wins = int(bets_df['pred_correct'].sum()) if n_bets>0 else 0
    losses = n_bets - wins
    final_capital = capital
    # Use tracked totals for staked/profit
    # total_staked and total_profit were already tracked incrementally
    if total_staked > 0:
        roi = (total_profit / total_staked) * 100
    else:
        roi = 0.0

    print(f"[{label}] Bets placed: {n_bets}")
    print(f"[{label}] Wins: {wins}, Losses: {losses}")
    print(f"[{label}] Final capital: {final_capital:.2f} EUR")
    print(f"[{label}] Total staked: {total_staked:.2f} EUR, Total profit: {total_profit:.2f} EUR, ROI on staked: {roi:.2f}%")
    print(f"[{label}] Candidate evaluations: {candidate_count}, skipped: {candidate_skipped}")

    # plotting: recreate plots from scratch
    try:
        plt.style.use('seaborn-darkgrid')
    except Exception:
        try:
            plt.style.use('seaborn')
        except Exception:
            plt.style.use('ggplot')

    x = df['match_index'].tolist()

    # defensive lengths
    L = len(x)
    ch = capital_history if len(capital_history)==L else (capital_history + [capital]*(L-len(capital_history)))
    sh = staked_history if len(staked_history)==L else (staked_history + [total_staked]*(L-len(staked_history)))
    ph = profit_history if len(profit_history)==L else (profit_history + [total_profit]*(L-len(profit_history)))
    ah = accuracy_history if len(accuracy_history)==L else (accuracy_history + [accuracy_history[-1] if accuracy_history else 0]*(L-len(accuracy_history)))
    bh = bets_count_history if len(bets_count_history)==L else (bets_count_history + [bets_placed_count]*(L-len(bets_count_history)))

    invested_value = [s + p for s, p in zip(sh, ch)]

    # (no debug prints)

    # Single-row plot: left y-axis for EUR (Staked and Staked+Profit), right y-axis for Accuracy (%)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot cumulative staked and staked+profit
    ax1.plot(x, sh, label='Staked (€)', color='tab:blue', linestyle='--')
    ax1.plot(x, invested_value, label='Staked + Profit (€)', color='tab:orange')

    # Mark bet events with red points using capital_after from the bets dataframe
    # this ensures losses are shown as drops (capital_after decreases)
    if n_bets > 0:
        bet_idx = bets_df['match_index'].tolist()
        bet_y = []
        for idx in bet_idx:
            try:
                cap_after = float(bets_df.loc[bets_df['match_index'] == int(idx), 'capital_after'].iloc[0])
            except Exception:
                # fallback to invested_value if something is missing
                i = int(idx) - 1
                cap_after = invested_value[i] if 0 <= i < len(invested_value) else None
            bet_y.append(cap_after)
        ax1.scatter(bet_idx, bet_y, color='red', s=40, marker='o', label='Bet (capital_after)')

    ax1.set_ylabel('EUR')
    ax1.set_xlabel('Match index (chronological)')
    ax1.set_title(f"Betting simulation ({label}): stake=€{stake}, threshold={threshold}x fair, min_fair={min_fair}")

    # Right axis: accuracy in percent
    ax2 = ax1.twinx()
    ax2.plot(x, [a * 100 for a in ah], label='Accuratezza (%)', color='tab:purple', linewidth=1.5)
    ax2.set_ylabel('Accuratezza (%)')

    # Optionally show cumulative bets as a faint step on left axis (helpful but not required)
    try:
        ax1.step(x, bh, where='post', color='tab:gray', alpha=0.25, label='Cumulative Bets (count)')
    except Exception:
        pass

    # Combined legend (handles from both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')

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
