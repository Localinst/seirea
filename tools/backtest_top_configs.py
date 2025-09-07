import pandas as pd
from pathlib import Path
import numpy as np

ART = Path(__file__).resolve().parents[1] / 'artifacts'
SIM = ART / 'simulation_evaluation.csv'
GRID = ART / 'betting_threshold_grid.csv'
GRID_UNW = ART / 'betting_threshold_grid_unweighted.csv'

def load_sim():
    df = pd.read_csv(SIM, parse_dates=['Date'], dayfirst=True)
    if 'DateTime_sort' in df.columns:
        df = df.sort_values(['DateTime_sort','Date'])
    else:
        df = df.sort_values('Date')
    df = df.reset_index(drop=True)
    return df

def backtest(df, multiplier, min_prob, min_fair=1.4, stake=1.0):
    capital = 0.0
    total_staked = 0.0
    total_profit = 0.0
    n_bets = 0
    wins = 0
    for _, row in df.iterrows():
        try:
            fair = float(row.get('fair_odds'))
            maxb = float(row.get('max_bookie_odds'))
            prob = float(row.get('predicted_pct'))
        except Exception:
            continue
        if pd.isna(fair) or pd.isna(maxb) or pd.isna(prob):
            continue
        if prob < min_prob:
            continue
        if fair < min_fair:
            continue
        if maxb >= fair * multiplier:
            n_bets += 1
            total_staked += stake
            pred_correct = bool(row.get('correct'))
            if pred_correct:
                profit = stake * (maxb - 1.0)
                wins += 1
            else:
                profit = -stake
            capital += profit
            total_profit += profit
    roi = (total_profit / total_staked * 100) if total_staked>0 else 0.0
    winrate = (wins / n_bets * 100) if n_bets>0 else 0.0
    return {
        'multiplier': multiplier,
        'min_prob': min_prob,
        'min_fair': min_fair,
        'n_bets': n_bets,
        'wins': wins,
        'winrate_pct': winrate,
        'total_staked': total_staked,
        'total_profit': total_profit,
        'roi_pct': roi,
        'final_capital': capital
    }

def pick_top_configs(grid_path, top_n=10, min_bets=30):
    if not grid_path.exists():
        return []
    g = pd.read_csv(grid_path)
    # prefer configurations with n_bets >= min_bets, otherwise relax
    g_ok = g[g['n_bets'] >= min_bets]
    if g_ok.empty:
        g_ok = g
    # sort by roi then final_capital
    g_ok = g_ok.sort_values(['roi_pct','final_capital'], ascending=False)
    return g_ok.head(top_n)

def main():
    df = load_sim()
    top_weighted = pick_top_configs(GRID, top_n=10, min_bets=30)
    top_unw = pick_top_configs(GRID_UNW, top_n=10, min_bets=30)

    configs = []
    # weighted grid entries may not contain min_fair; use 1.4 default
    if not top_weighted.empty:
        for _, r in top_weighted.iterrows():
            configs.append((float(r['multiplier']), float(r['min_prob']), 1.4, 'weighted'))
    if not top_unw.empty:
        for _, r in top_unw.iterrows():
            min_fair = float(r.get('min_fair', 1.4))
            configs.append((float(r['multiplier']), float(r['min_prob']), min_fair, 'unweighted'))

    results = []
    for m, p, mf, tag in configs:
        res = backtest(df, m, p, min_fair=mf)
        res['source'] = tag
        results.append(res)

    out = ART / 'backtest_top_configs_results.csv'
    pd.DataFrame(results).to_csv(out, index=False)
    print(f'Saved backtest results to {out}')
    # print top 10 by roi
    dfres = pd.DataFrame(results)
    dfres = dfres.sort_values('roi_pct', ascending=False)
    print('Top backtest results:')
    print(dfres.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
