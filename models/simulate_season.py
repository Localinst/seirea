"""Simulate season 2024-2025 by predicting every match in chronological order.

For each match the script:
- computes pre-match features using historical matches only (reuses compute_on_the_fly from predict/predict_next.py),
- loads the stacked calibrated model if available (falls back to stacked),
- predicts probabilities and the most likely outcome,
- saves a CSV with Date, HomeTeam, AwayTeam, prob_home, prob_draw, prob_away, predicted_label, predicted_pct.

Run from project root:
  python models\simulate_season.py
"""
import os
import sys
import pandas as pd
from datetime import datetime
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(__file__))
COMBINED = os.path.join(ROOT, 'data', 'combined_matches.csv')
FEATURES = os.path.join(ROOT, 'data', 'features.csv')
ART = os.path.join(ROOT, 'artifacts')
STACKED_CAL = os.path.join(ART, 'stacked_calibrated_model.joblib')
STACKED = os.path.join(ART, 'stacked_model.joblib')
# We'll write one output per season
OUT_CSV_TMPL = os.path.join(ART, 'simulation_{season}_predictions.csv')
LEAKAGE_REPORT_TMPL = os.path.join(ART, 'leakage_report_{season}.txt')

# Ensure repo root on path so we can import predict/predict_next
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print('Loading helper module predict/predict_next...')
import predict.predict_next as pn
import joblib

# prefer calibrated stacked
if os.path.exists(STACKED_CAL):
    stack_path = STACKED_CAL
    used_calibrated = True
elif os.path.exists(STACKED):
    stack_path = STACKED
    used_calibrated = False
else:
    raise RuntimeError('No stacked artifact found in artifacts/; run training first')

print('Loading model artifact from', stack_path)
model_obj, stored_features = pn.load_artifact(stack_path)

# determine model feature columns
# load a small features df to get numeric fallback
# read features file (for leakage checks)
if os.path.exists(FEATURES):
    df_feats = pd.read_csv(FEATURES, parse_dates=['Date'], dayfirst=True)
else:
    df_feats = pd.DataFrame()
all_num = [c for c in df_feats.columns if pd.api.types.is_numeric_dtype(df_feats[c])] if not df_feats.empty else []
model_feat = pn.resolve_feature_cols_for_model(model_obj, stored_features, all_num)
print('Model will use', len(model_feat), 'feature columns')

# load combined matches once
comb = pd.read_csv(COMBINED, parse_dates=['Date'], dayfirst=True)

# seasons to simulate: (start, end, season_key)
seasons = [
    ('2023-07-01', '2024-06-30', '2023_2024'),
    ('2024-07-01', '2025-06-30', '2024_2025'),
]

os.makedirs(ART, exist_ok=True)

for s_start, s_end, s_key in seasons:
    print(f'Starting simulation for season {s_key}...')
    start = pd.to_datetime(s_start)
    end = pd.to_datetime(s_end)
    season_df = comb[(comb['Date'] >= start) & (comb['Date'] <= end)].copy()
    season_df = season_df.sort_values('Date').reset_index(drop=True)
    print(f'Matches in {s_key} window:', len(season_df))

    results = []
    leakage_lines = []
    count = 0
    for _, r in season_df.iterrows():
        count += 1
        home = r['HomeTeam']
        away = r['AwayTeam']
        match_dt = r['Date'] if isinstance(r['Date'], pd.Timestamp) else pd.to_datetime(r['Date'])
        date = match_dt.strftime('%Y-%m-%d')
        print(f'[{count}/{len(season_df)}] Predicting {home} vs {away} on {date}...')

        # Check leakage in source files: combined and features
        try:
            future_comb = comb[comb['Date'] >= match_dt]
            future_feats = df_feats[df_feats['Date'] >= match_dt] if not df_feats.empty else pd.DataFrame()
            if len(future_comb) > 0 or len(future_feats) > 0:
                leakage_lines.append(f"{date} {home} vs {away}: combined_future_rows={len(future_comb)}, features_future_rows={len(future_feats)}")
        except Exception as e:
            leakage_lines.append(f"{date} {home} vs {away}: leakage check error: {e}")

        # To avoid leakage, create a filtered temporary combined file with only past matches
        try:
            filtered_comb = comb[comb['Date'] < match_dt]
            tmpf = None
            if len(filtered_comb) == 0:
                # no historical data -> still attempt predict_on_the_fly with empty combined
                tmpf = tempfile.NamedTemporaryFile(delete=False, dir=ART, suffix='.csv')
                filtered_comb.to_csv(tmpf.name, index=False)
                tmpf.close()
                comb_path_for_call = tmpf.name
            else:
                tmpf = tempfile.NamedTemporaryFile(delete=False, dir=ART, suffix='.csv')
                filtered_comb.to_csv(tmpf.name, index=False)
                tmpf.close()
                comb_path_for_call = tmpf.name

            # compute pre-match features using historical matches only (filtered file)
            try:
                rowd = pn.compute_on_the_fly(home, away, date, comb_path_for_call)
                X = pd.DataFrame([rowd])
                Xsub = X.reindex(columns=model_feat).fillna(0.0)
                probs = pn.predict_proba_safe(model_obj, Xsub, model_feat)[0]
            except Exception as e:
                print('  Error predicting match:', e)
                traceback.print_exc()
                probs = [None, None, None]
        finally:
            # cleanup temp file
            try:
                if 'tmpf' in locals() and tmpf is not None:
                    os.unlink(tmpf.name)
            except Exception:
                pass

        if probs[0] is None:
            pred_label = None
            pred_pct = None
        else:
            lab_idx = int(pd.np.argmax(probs)) if hasattr(pd, 'np') else int(__import__('numpy').argmax(probs))
            labels = {0: 'Home', 1: 'Draw', 2: 'Away'}
            pred_label = labels.get(lab_idx, '')
            pred_pct = float(probs[lab_idx])

        results.append({
            'Date': date,
            'HomeTeam': home,
            'AwayTeam': away,
            'prob_home': float(probs[0]) if probs[0] is not None else None,
            'prob_draw': float(probs[1]) if probs[1] is not None else None,
            'prob_away': float(probs[2]) if probs[2] is not None else None,
            'predicted': pred_label,
            'predicted_pct': pred_pct,
        })

    # save to CSV and leakage report for this season
    out_csv = OUT_CSV_TMPL.format(season=s_key)
    out_report = LEAKAGE_REPORT_TMPL.format(season=s_key)
    outdf = pd.DataFrame(results)
    outdf.to_csv(out_csv, index=False)
    with open(out_report, 'w', encoding='utf-8') as fh:
        if len(leakage_lines) == 0:
            fh.write('No leakage detected (no future-dated rows found relative to match dates)\n')
        else:
            fh.write('\n'.join(leakage_lines))

    print('Wrote predictions to', out_csv)
    print('Wrote leakage report to', out_report)
