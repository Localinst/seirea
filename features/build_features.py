"""Build richer features from cleaned matches.

Features added:
- last-N rolling aggregates (pts, goal diff)
- season-to-date aggregates per team (pts, wins, draws, losses, gf, ga, shots)
- head-to-head last-K stats
- days since last match for each team
- Elo ratings updated iteratively (home advantage included)

All features are computed using only past matches (no leakage).
Saves features to `data/features.csv`.
"""

import pandas as pd
import os
from collections import deque, defaultdict
import math

ROOT = os.path.dirname(os.path.dirname(__file__))
IN = os.path.join(ROOT, 'data', 'combined_matches.csv')
OUT = os.path.join(ROOT, 'data', 'features.csv')


def extract_season(srcfile):
    # expects filename like season-1718.csv -> returns '1718'
    try:
        base = os.path.basename(srcfile)
        if base.startswith('season-'):
            return base.split('-')[1].split('.')[0]
    except Exception:
        pass
    return None


def compute_features(df, rolling_n=5, h2h_k=5, elo_k=20, home_adv=100):
    df = df.sort_values('Date').copy()

    # per-team structures
    team_season_stats = defaultdict(lambda: {'season': None, 'pts': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0, 'shots': 0, 'matches': 0})
    # keep up to 10 for moving averages 3/5/10
    team_recent = defaultdict(lambda: {'pts': deque(maxlen=10), 'gd': deque(maxlen=10), 'shots': deque(maxlen=10), 'goals': deque(maxlen=10)})
    last_match_date = {}
    elo = defaultdict(lambda: 1500.0)

    # head-to-head: key as tuple(sorted([a,b])) -> deque of (home_team, home_goals, away_goals)
    h2h = defaultdict(lambda: deque(maxlen=h2h_k))

    rows = []

    for _, r in df.iterrows():
        home = r['HomeTeam']
        away = r['AwayTeam']
        date = r['Date']
        src = r.get('source_file', '')
        season = extract_season(src)

        # reset season stats if season changed
        if team_season_stats[home]['season'] != season:
            team_season_stats[home] = {'season': season, 'pts': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0, 'shots': 0, 'matches': 0}
        if team_season_stats[away]['season'] != season:
            team_season_stats[away] = {'season': season, 'pts': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0, 'shots': 0, 'matches': 0}

        # features to compute BEFORE updating stats with current match
        def team_features(team):
            s = team_season_stats[team]
            rec = team_recent[team]
            last_date = last_match_date.get(team)
            days_since = (date - last_date).days if last_date is not None else -1
            recent_pts_mean = float(sum(rec['pts'])/len(rec['pts'])) if len(rec['pts'])>0 else 0.0
            recent_gd_mean = float(sum(rec['gd'])/len(rec['gd'])) if len(rec['gd'])>0 else 0.0
            recent_shots_sum = float(sum(rec['shots'])) if len(rec['shots'])>0 else 0.0
            recent_goals_sum = float(sum(rec['goals'])) if len(rec['goals'])>0 else 0.0
            recent_conversion = (recent_goals_sum / recent_shots_sum) if recent_shots_sum>0 else 0.0
            # moving averages for points for windows 3,5,10
            def ma(a, n):
                if len(a)==0:
                    return 0.0
                return float(sum(list(a)[-n:]) / min(len(a), n))
            recent_pts_ma_3 = ma(rec['pts'], 3)
            recent_pts_ma_5 = ma(rec['pts'], 5)
            recent_pts_ma_10 = ma(rec['pts'], 10)
            return {
                'season_pts': s['pts'], 'season_wins': s['wins'], 'season_draws': s['draws'], 'season_losses': s['losses'],
                'season_gf': s['gf'], 'season_ga': s['ga'], 'season_matches': s['matches'], 'season_avg_shots': (s['shots']/s['matches'] if s['matches']>0 else 0.0),
                'recent_pts_mean': recent_pts_mean, 'recent_gd_mean': recent_gd_mean, 'recent_matches': len(rec['pts']), 'days_since_last': days_since, 'days_since_log': (math.log1p(days_since) if days_since>=0 else -1),
                'recent_shots_sum': recent_shots_sum, 'recent_goals_sum': recent_goals_sum, 'recent_conversion': recent_conversion,
                'recent_pts_ma_3': recent_pts_ma_3, 'recent_pts_ma_5': recent_pts_ma_5, 'recent_pts_ma_10': recent_pts_ma_10,
                'elo': elo[team]
            }

        home_stats = team_features(home)
        away_stats = team_features(away)

        # head-to-head stats
        pair = tuple(sorted([home, away]))
        hdeque = h2h[pair]
        h2h_home_goals_avg = 0.0
        h2h_away_goals_avg = 0.0
        h2h_home_pts_mean = 0.0
        if len(hdeque) > 0:
            # compute from stored records
            hg = []
            ag = []
            pts_home = []
            for rec_h, hg_val, ag_val in hdeque:
                # rec_h is the home team in the historical record
                if rec_h == home:
                    hg.append(hg_val); ag.append(ag_val)
                    if hg_val>ag_val: pts_home.append(3)
                    elif hg_val==ag_val: pts_home.append(1)
                    else: pts_home.append(0)
                else:
                    # historical home was the other team; invert
                    hg.append(ag_val); ag.append(hg_val)
                    if ag_val>hg_val: pts_home.append(3)
                    elif ag_val==hg_val: pts_home.append(1)
                    else: pts_home.append(0)
            h2h_home_goals_avg = float(sum(hg)/len(hg))
            h2h_away_goals_avg = float(sum(ag)/len(ag))
            h2h_home_pts_mean = float(sum(pts_home)/len(pts_home))

        row = r.to_dict()
        # Remove raw post-match columns from the output to avoid leakage.
        # We still use r[...] below to update internal state, but we don't
        # expose these in the saved features file.
        postmatch_cols = ['FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
        for c in postmatch_cols:
            row.pop(c, None)

        # add team stats prefixed
        for k, v in home_stats.items():
            row['home_' + k] = v
        for k, v in away_stats.items():
            row['away_' + k] = v

        # head-to-head features
        row['h2h_home_goals_avg'] = h2h_home_goals_avg
        row['h2h_away_goals_avg'] = h2h_away_goals_avg
        row['h2h_home_pts_mean'] = h2h_home_pts_mean
        # elo difference
        row['elo_diff'] = home_stats['elo'] - away_stats['elo']

        rows.append(row)

        # now update stats with this match outcome (must be inside the loop)
        ft_h = r.get('FTHG', 0) if not pd.isna(r.get('FTHG', 0)) else 0
        ft_a = r.get('FTAG', 0) if not pd.isna(r.get('FTAG', 0)) else 0

        # points
        if ft_h > ft_a:
            home_pts, away_pts = 3, 0
            team_season_stats[home]['wins'] += 1
            team_season_stats[away]['losses'] += 1
        elif ft_h < ft_a:
            home_pts, away_pts = 0, 3
            team_season_stats[away]['wins'] += 1
            team_season_stats[home]['losses'] += 1
        else:
            home_pts, away_pts = 1, 1
            team_season_stats[home]['draws'] += 1
            team_season_stats[away]['draws'] += 1

        team_season_stats[home]['pts'] += home_pts
        team_season_stats[away]['pts'] += away_pts
        team_season_stats[home]['gf'] += ft_h
        team_season_stats[home]['ga'] += ft_a
        team_season_stats[away]['gf'] += ft_a
        team_season_stats[away]['ga'] += ft_h
        team_season_stats[home]['matches'] += 1
        team_season_stats[away]['matches'] += 1

        # shots if available
        for col, team in [('HS', home), ('AS', away)]:
            if col in r and not pd.isna(r[col]):
                try:
                    team_season_stats[team]['shots'] += int(r[col])
                except Exception:
                    pass

        # update recent deques (points, gd, shots, goals)
        team_recent[home]['pts'].append(home_pts)
        team_recent[home]['gd'].append(ft_h - ft_a)
        team_recent[home]['shots'].append(r.get('HS', 0) if not pd.isna(r.get('HS', 0)) else 0)
        team_recent[home]['goals'].append(ft_h)
        team_recent[away]['pts'].append(away_pts)
        team_recent[away]['gd'].append(ft_a - ft_h)
        team_recent[away]['shots'].append(r.get('AS', 0) if not pd.isna(r.get('AS', 0)) else 0)
        team_recent[away]['goals'].append(ft_a)

        # update last match date
        last_match_date[home] = date
        last_match_date[away] = date

        # update head-to-head deque
        h2h[pair].append((home, ft_h, ft_a))

        # update elo ratings
        # Elo expected using logistic approximation
        def expected_score(rating_a, rating_b, home_adv_local=0.0):
            diff = rating_a - rating_b + home_adv_local
            return 1.0 / (1.0 + 10 ** (-diff / 400.0))

        ea = expected_score(elo[home], elo[away], home_adv)
        eb = expected_score(elo[away], elo[home], -home_adv)
        # actual scores
        if ft_h > ft_a:
            sa, sb = 1.0, 0.0
        elif ft_h < ft_a:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        elo[home] += elo_k * (sa - ea)
        elo[away] += elo_k * (sb - eb)

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(IN, parse_dates=['Date'])
    feats = compute_features(df, rolling_n=5, h2h_k=5, elo_k=20, home_adv=100)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    feats.to_csv(OUT, index=False)
    print('Saved', OUT)


if __name__ == '__main__':
    main()
