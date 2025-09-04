#!/usr/bin/env python3
"""CLI to predict outcome probabilities for a new match using the chosen best model (stacked).

Usage examples:
  python predict/predict_next.py --home Bologna --away Genoa --date 2024-01-05

The script looks up `data/features.csv` for a matching row (Date, HomeTeam, AwayTeam).
If found, it will subset features to what the saved model expects and print a readable report
with probabilities expressed as percentages.
"""
import os
import argparse
import joblib
import json
import pandas as pd
import numpy as np
import sys
import importlib.util
import math
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
ART = os.path.join(ROOT, 'artifacts')
STACKED_PATH = os.path.join(ART, 'stacked_model.joblib')
STACKED_CAL_PATH = os.path.join(ART, 'stacked_calibrated_model.joblib')


def load_artifact(path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        model = obj.get('model', None)
        features = obj.get('features', None)
    else:
        model = obj
        features = None
    return model, features


def resolve_feature_cols_for_model(model, stored_feat, global_feat_cols):
    if stored_feat:
        return stored_feat
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return list(global_feat_cols)


def predict_proba_safe(model_obj, X, model_feat_cols):
    Xsub = X.reindex(columns=model_feat_cols).fillna(0.0)
    try:
        return model_obj.predict_proba(Xsub)
    except Exception:
        try:
            return model_obj.predict_proba(Xsub.values)
        except Exception as e:
            raise RuntimeError(f"Can't call predict_proba on model: {e}")
def compute_on_the_fly(home, away, date, combined_path, seed_path=None):
    """Compute feature row for a future match (home vs away on date) using past matches only.
    Returns a dict with the same prefixed keys as compute_features() in features/build_features.py
    """
    # allow using either the main combined file, an optional seed file, or both
    parts = []
    if combined_path and os.path.exists(combined_path):
        parts.append(pd.read_csv(combined_path, parse_dates=['Date']))
    if seed_path:
        if os.path.exists(seed_path):
            parts.append(pd.read_csv(seed_path, parse_dates=['Date']))
        else:
            raise FileNotFoundError(f"Seed results file not found at {seed_path}")
    if len(parts) == 0:
        # No historical matches provided; proceed with empty history so we can still
        # compute default features (zeros / default elo). This enables early-season
        # predictions when no past-match CSV is available.
        print('Warning: no combined_matches.csv or seed provided — computing default features with empty history')
        df = pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam'])
    else:
        df = pd.concat(parts, ignore_index=True)
    # drop exact duplicate matches (Date, HomeTeam, AwayTeam) keeping first occurrence
    if 'Date' in df.columns and 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first')
    df = df.sort_values('Date')
    # structures copied from features/build_features.py
    from collections import deque, defaultdict

    team_season_stats = defaultdict(lambda: {'season': None, 'pts': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0, 'shots': 0, 'matches': 0})
    team_recent = defaultdict(lambda: {'pts': deque(maxlen=10), 'gd': deque(maxlen=10), 'shots': deque(maxlen=10), 'goals': deque(maxlen=10)})
    last_match_date = {}
    elo = defaultdict(lambda: 1500.0)
    h2h = defaultdict(lambda: deque(maxlen=5))

    def extract_season(srcfile):
        try:
            base = os.path.basename(srcfile)
            if base.startswith('season-'):
                return base.split('-')[1].split('.')[0]
        except Exception:
            pass
        return None

    def expected_score(rating_a, rating_b, home_adv_local=0.0):
        diff = rating_a - rating_b + home_adv_local
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    match_date = pd.to_datetime(date)

    for _, r in df.iterrows():
        if pd.to_datetime(r['Date']) >= match_date:
            break
        home_t = r['HomeTeam']
        away_t = r['AwayTeam']
        src = r.get('source_file', '')
        season = extract_season(src)
        # reset season stats if changed
        if team_season_stats[home_t]['season'] != season:
            team_season_stats[home_t] = {'season': season, 'pts': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0, 'shots': 0, 'matches': 0}
        if team_season_stats[away_t]['season'] != season:
            team_season_stats[away_t] = {'season': season, 'pts': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0, 'shots': 0, 'matches': 0}

        # update with this historical match
        ft_h = r.get('FTHG', 0) if not pd.isna(r.get('FTHG', 0)) else 0
        ft_a = r.get('FTAG', 0) if not pd.isna(r.get('FTAG', 0)) else 0
        if ft_h > ft_a:
            home_pts, away_pts = 3, 0
            team_season_stats[home_t]['wins'] += 1
            team_season_stats[away_t]['losses'] += 1
        elif ft_h < ft_a:
            home_pts, away_pts = 0, 3
            team_season_stats[away_t]['wins'] += 1
            team_season_stats[home_t]['losses'] += 1
        else:
            home_pts, away_pts = 1, 1
            team_season_stats[home_t]['draws'] += 1
            team_season_stats[away_t]['draws'] += 1

        team_season_stats[home_t]['pts'] += home_pts
        team_season_stats[away_t]['pts'] += away_pts
        team_season_stats[home_t]['gf'] += ft_h
        team_season_stats[home_t]['ga'] += ft_a
        team_season_stats[away_t]['gf'] += ft_a
        team_season_stats[away_t]['ga'] += ft_h
        team_season_stats[home_t]['matches'] += 1
        team_season_stats[away_t]['matches'] += 1
        # shots
        try:
            hs = int(r.get('HS', 0)) if not pd.isna(r.get('HS', 0)) else 0
            as_ = int(r.get('AS', 0)) if not pd.isna(r.get('AS', 0)) else 0
        except Exception:
            hs, as_ = 0, 0
        team_season_stats[home_t]['shots'] += hs
        team_season_stats[away_t]['shots'] += as_

        team_recent[home_t]['pts'].append(home_pts)
        team_recent[home_t]['gd'].append(ft_h - ft_a)
        team_recent[home_t]['shots'].append(hs)
        team_recent[home_t]['goals'].append(ft_h)
        team_recent[away_t]['pts'].append(away_pts)
        team_recent[away_t]['gd'].append(ft_a - ft_h)
        team_recent[away_t]['shots'].append(as_)
        team_recent[away_t]['goals'].append(ft_a)

        last_match_date[home_t] = pd.to_datetime(r['Date'])
        last_match_date[away_t] = pd.to_datetime(r['Date'])

        # head to head
        pair = tuple(sorted([home_t, away_t]))
        h2h[pair].append((home_t, ft_h, ft_a))

        # elo update
        ea = expected_score(elo[home_t], elo[away_t], 100.0)
        eb = expected_score(elo[away_t], elo[home_t], -100.0)
        if ft_h > ft_a:
            sa, sb = 1.0, 0.0
        elif ft_h < ft_a:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        elo[home_t] += 20 * (sa - ea)
        elo[away_t] += 20 * (sb - eb)

    # helper to compute team features at match_date
    def team_features(team):
        s = team_season_stats[team]
        rec = team_recent[team]
        last_date = last_match_date.get(team)
        days_since = (match_date - last_date).days if last_date is not None else -1
        recent_pts_mean = float(sum(rec['pts']) / len(rec['pts'])) if len(rec['pts']) > 0 else 0.0
        recent_gd_mean = float(sum(rec['gd']) / len(rec['gd'])) if len(rec['gd']) > 0 else 0.0
        recent_shots_sum = float(sum(rec['shots'])) if len(rec['shots']) > 0 else 0.0
        recent_goals_sum = float(sum(rec['goals'])) if len(rec['goals']) > 0 else 0.0
        recent_conversion = (recent_goals_sum / recent_shots_sum) if recent_shots_sum > 0 else 0.0
        def ma(a, n):
            if len(a) == 0:
                return 0.0
            return float(sum(list(a)[-n:]) / min(len(a), n))
        recent_pts_ma_3 = ma(rec['pts'], 3)
        recent_pts_ma_5 = ma(rec['pts'], 5)
        recent_pts_ma_10 = ma(rec['pts'], 10)
        return {
            'season_pts': s['pts'], 'season_wins': s['wins'], 'season_draws': s['draws'], 'season_losses': s['losses'],
            'season_gf': s['gf'], 'season_ga': s['ga'], 'season_matches': s['matches'], 'season_avg_shots': (s['shots'] / s['matches'] if s['matches'] > 0 else 0.0),
            'recent_pts_mean': recent_pts_mean, 'recent_gd_mean': recent_gd_mean, 'recent_matches': len(rec['pts']), 'days_since_last': days_since, 'days_since_log': (math.log1p(days_since) if days_since >= 0 else -1),
            'recent_shots_sum': recent_shots_sum, 'recent_goals_sum': recent_goals_sum, 'recent_conversion': recent_conversion,
            'recent_pts_ma_3': recent_pts_ma_3, 'recent_pts_ma_5': recent_pts_ma_5, 'recent_pts_ma_10': recent_pts_ma_10,
            'elo': elo[team]
        }

    home_stats = team_features(home)
    away_stats = team_features(away)
    pair = tuple(sorted([home, away]))
    hdeque = h2h[pair]
    h2h_home_goals_avg = 0.0
    h2h_away_goals_avg = 0.0
    h2h_home_pts_mean = 0.0
    if len(hdeque) > 0:
        hg = []
        ag = []
        pts_home = []
        for rec_h, hg_val, ag_val in hdeque:
            if rec_h == home:
                hg.append(hg_val); ag.append(ag_val)
                if hg_val > ag_val: pts_home.append(3)
                elif hg_val == ag_val: pts_home.append(1)
                else: pts_home.append(0)
            else:
                hg.append(ag_val); ag.append(hg_val)
                if ag_val > hg_val: pts_home.append(3)
                elif ag_val == hg_val: pts_home.append(1)
                else: pts_home.append(0)
        h2h_home_goals_avg = float(sum(hg) / len(hg))
        h2h_away_goals_avg = float(sum(ag) / len(ag))
        h2h_home_pts_mean = float(sum(pts_home) / len(pts_home))

    row = {}
    for k, v in home_stats.items():
        row['home_' + k] = v
    for k, v in away_stats.items():
        row['away_' + k] = v
    row['h2h_home_goals_avg'] = h2h_home_goals_avg
    row['h2h_away_goals_avg'] = h2h_away_goals_avg
    row['h2h_home_pts_mean'] = h2h_home_pts_mean
    row['elo_diff'] = home_stats['elo'] - away_stats['elo']
    # add basic identifiers
    row['Date'] = match_date
    row['HomeTeam'] = home
    row['AwayTeam'] = away
    return row


def interpret_probs(probs):
    labels = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    idx = int(np.argmax(probs))
    pct = probs[idx] * 100.0
    second = np.partition(probs, -2)[-2]
    margin = probs[idx] - second
    return labels[idx], pct, margin * 100.0


def main():
    p = argparse.ArgumentParser(description='Predict match outcome probabilities (Home/Draw/Away)')
    p.add_argument('--home', required=True, help='Home team name as in features.csv')
    p.add_argument('--away', required=True, help='Away team name as in features.csv')
    p.add_argument('--date', required=True, help='Match date YYYY-MM-DD')
    p.add_argument('--seed-results', required=False, help='Optional CSV with early-season results to seed historical matches for on-the-fly feature computation')
    p.add_argument('--show-json', action='store_true', help='Also print raw JSON result')
    args = p.parse_args()

    if not os.path.exists(FEATS):
        print('features.csv not found at', FEATS)
        sys.exit(2)

    df = pd.read_csv(FEATS, parse_dates=['Date'])
    # find exact match row
    try:
        date = pd.to_datetime(args.date).date()
    except Exception:
        print('Invalid date format, use YYYY-MM-DD')
        sys.exit(2)

    mask = (df['Date'].dt.date == date) & (df['HomeTeam'].str.lower() == args.home.lower()) & (df['AwayTeam'].str.lower() == args.away.lower())
    if not mask.any():
        print(f'No matching row found for {args.home} vs {args.away} on {args.date} in {FEATS}. Computing features on-the-fly from historical matches...')
        # compute from combined matches (future match scenario)
        COMBINED = os.path.join(os.path.dirname(ROOT), 'data', 'combined_matches.csv')
        try:
            rowd = compute_on_the_fly(args.home, args.away, args.date, COMBINED, seed_path=args.seed_results)
            row = pd.Series(rowd)
        except FileNotFoundError as e:
            print('Cannot compute on-the-fly features:', e)
            print('Possible fixes: generate features.csv including the target match, or provide combined_matches.csv')
            sys.exit(3)
        except Exception as e:
            print('Error computing features on-the-fly:', e)
            sys.exit(3)
    else:
        row = df[mask].iloc[0]

    # prefer calibrated stacked if available
    if os.path.exists(STACKED_CAL_PATH):
        stack_path = STACKED_CAL_PATH
        used_calibrated = True
    else:
        stack_path = STACKED_PATH
        used_calibrated = False

    if not os.path.exists(stack_path):
        print('Stacked model artifact not found at', stack_path)
        sys.exit(4)
    # Ensure pickled helper classes are importable for unpickling (StackedModel / CalibratedStacked)
    try:
        import models.tune_and_stack as _ts
        try:
            import __main__ as _m
            setattr(_m, 'StackedModel', getattr(_ts, 'StackedModel'))
        except Exception:
            pass
    except Exception:
        fn = os.path.join(os.path.dirname(__file__), 'models', 'tune_and_stack.py')
        fn2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'tune_and_stack.py')
        fn_use = fn if os.path.exists(fn) else (fn2 if os.path.exists(fn2) else None)
        if fn_use:
            spec = importlib.util.spec_from_file_location('models.tune_and_stack', fn_use)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['models.tune_and_stack'] = mod
            try:
                import __main__ as _m
                setattr(_m, 'StackedModel', getattr(mod, 'StackedModel'))
            except Exception:
                pass

    try:
        import models.calibrate_stacked_oof as _cal
        try:
            import __main__ as _m
            if hasattr(_cal, 'CalibratedStacked'):
                setattr(_m, 'CalibratedStacked', getattr(_cal, 'CalibratedStacked'))
        except Exception:
            pass
    except Exception:
        # ignore if module missing; calibrated artifact may still be pickleable if created in this env
        pass

    # load artifact, but handle pickling errors when class was defined under __main__ previously
    try:
        model_obj, stored_features = load_artifact(stack_path)
    except Exception as e:
        # try to import helper modules and expose expected classes to __main__, then retry
        try:
            import models.calibrate_stacked_oof as _cal
            import __main__ as _m
            if hasattr(_cal, 'CalibratedStacked'):
                setattr(_m, 'CalibratedStacked', getattr(_cal, 'CalibratedStacked'))
        except Exception:
            pass
        try:
            import models.tune_and_stack as _ts
            import __main__ as _m
            if hasattr(_ts, 'StackedModel'):
                setattr(_m, 'StackedModel', getattr(_ts, 'StackedModel'))
        except Exception:
            pass

        try:
            model_obj, stored_features = load_artifact(stack_path)
        except Exception as e2:
            print('Could not load calibrated artifact (unpickle failure). Falling back to uncalibrated stacked model. Error:', e2)
            stack_path = STACKED_PATH
            used_calibrated = False
            model_obj, stored_features = load_artifact(stack_path)

    # determine global numeric feature fallback
    all_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ('y', 'Result')]
    model_feat = resolve_feature_cols_for_model(model_obj, stored_features, all_num)

    # ensure single-row DataFrame for prediction
    if isinstance(row, pd.Series):
        X = pd.DataFrame([row.to_dict()])
    else:
        X = pd.DataFrame([row])
    X = X.reindex(columns=model_feat).fillna(0.0)

    probs = predict_proba_safe(model_obj, X, model_feat)[0]

    label, pct, margin = interpret_probs(probs)

    out = {
        'date': str(date),
        'home': args.home,
        'away': args.away,
        'model_used': 'stacked_calibrated' if used_calibrated else 'stacked',
        'probabilities': {'home_win': float(probs[0]), 'draw': float(probs[1]), 'away_win': float(probs[2])},
        'predicted': label,
        'predicted_pct': round(pct, 2),
        'margin_pct': round(margin, 2)
    }

    print('Match:', f"{args.home} vs {args.away} on {date}")
    print('Model: stacked (chosen as best based on earlier evaluations)')
    print('\nProbabilities:')
    print(f"  Home win : {probs[0]*100:.2f}%")
    print(f"  Draw     : {probs[1]*100:.2f}%")
    print(f"  Away win : {probs[2]*100:.2f}%")
    print('\nPrediction:')
    print(f"  {label} — confidence {pct:.2f}% (margin vs 2nd: {margin:.2f} percentage points)")

    # quick interpretation hints
    if pct < 55:
        print('\nNote: low confidence (<55%). Consider this a weak signal; check calibration plots in artifacts/.')
    elif pct < 65:
        print('\nNote: moderate confidence. Probabilities are useful but not decisive.')
    else:
        print('\nNote: high confidence — model leans strongly toward this outcome.')

    if args.show_json:
        print('\nRaw JSON:')
        print(json.dumps(out, indent=2))

    # Save a small probability bar chart into artifacts
    try:
        os.makedirs(ART, exist_ok=True)
        labs = ['Home win', 'Draw', 'Away win']
        vals = [probs[0]*100, probs[1]*100, probs[2]*100]
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(labs, vals, color=['#2ca02c', '#1f77b4', '#d62728'])
        ax.set_ylim(0, 100)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha='center')
        ax.set_ylabel('Probability (%)')
        ax.set_title(f"Predicted probabilities {args.home} vs {args.away} {date}")
        plt.tight_layout()
        fname = f'pred_prob_{args.home}_{args.away}_{date}.png'.replace(' ', '_')
        chart_path = os.path.join(ART, fname)
        fig.savefig(chart_path)
        plt.close(fig)
        print('\nSaved probability chart to', chart_path)
    except Exception as e:
        print('Could not save chart:', e)

    # list available graphs that may have been generated by model training / calibration
    graphs = []
    try:
        for fn in os.listdir(ART):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')) and fn.startswith(('calibration_', 'prob_hist_', 'pred_prob_', 'calibration_stacked', 'prob_hist_stacked')):
                graphs.append(fn)
    except Exception:
        pass
    if graphs:
        print('\nGraphs available in artifacts/:')
        for g in sorted(graphs):
            print('  -', g)


if __name__ == '__main__':
    main()
