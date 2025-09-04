"""Compare baseline, lgb_calibrated and stacked models: per-season metrics, calibration curves,
probability histograms. Save tuning summary and run a single prediction for a 2024 match.
"""
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
OUT = os.path.join(ROOT, 'artifacts')
os.makedirs(OUT, exist_ok=True)

MODEL_PATHS = {
    'baseline': os.path.join(OUT, 'baseline_model.joblib'),
    'lgb_calibrated': os.path.join(OUT, 'lgb_calibrated.joblib'),
    'stacked': os.path.join(OUT, 'stacked_model.joblib')
}

TUNING_SUMMARY = os.path.join(OUT, 'tuning_summary.json')


def load_artifact(path):
    obj = joblib.load(path)
    # artifact might be dict {'model':..., 'features':...} or raw estimator
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
    # try pipeline steps
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    # fallback
    return list(global_feat_cols)


def predict_proba_safe(model_obj, X, model_feat_cols):
    # X: DataFrame; model_feat_cols: list of columns model expects
    # subset and reindex X to model's expectations
    Xsub = X.reindex(columns=model_feat_cols).fillna(0.0)
    try:
        return model_obj.predict_proba(Xsub)
    except Exception:
        try:
            return model_obj.predict_proba(Xsub.values)
        except Exception as e:
            raise RuntimeError(f"Can't call predict_proba on model: {e}")


def multiclass_brier(y_true, probs):
    # y_true: encoded 0..K-1, probs: n x K
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[y_true]
    return np.mean((probs - onehot) ** 2)


def main():
    # Ensure stacked model class is importable for unpickling
    try:
        import models.tune_and_stack as _ts
    except Exception:
        fn = os.path.join(os.path.dirname(__file__), 'tune_and_stack.py')
        if os.path.exists(fn):
            spec = importlib.util.spec_from_file_location('models.tune_and_stack', fn)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['models.tune_and_stack'] = mod
            # Also expose StackedModel under __main__ to handle pickles created when that
            # module was executed as a script (__main__). This makes unpickling robust.
            try:
                import __main__ as _m
                setattr(_m, 'StackedModel', getattr(mod, 'StackedModel'))
            except Exception:
                pass
    
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['y'] = df['Result'].map({'H': 0, 'D': 1, 'A': 2})

    # Load models
    models = {}
    features_map = {}
    for name, p in MODEL_PATHS.items():
        if os.path.exists(p):
            m, f = load_artifact(p)
            models[name] = m
            features_map[name] = f
        else:
            print('Missing model artifact:', p)

    # Determine a global numeric feature set (fallback)
    all_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ('y',)]
    global_feat_cols = all_num

    # Split train/test by date as used elsewhere
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']

    seasons = sorted(df['Date'].dt.year.unique())

    # Per-season metrics: for each model, compute log_loss, accuracy, brier
    rows = []
    for season in seasons:
        season_df = df[df['Date'].dt.year == season]
        if season_df.empty:
            continue
        y_true = season_df['y'].values
        for name, m in models.items():
            # resolve feature columns for this model
            model_feat = resolve_feature_cols_for_model(m, features_map.get(name), global_feat_cols)
            Xs = season_df.reindex(columns=model_feat).fillna(0.0)
            try:
                probs = predict_proba_safe(m, Xs, model_feat)
            except Exception:
                obj = joblib.load(MODEL_PATHS[name])
                model = obj.get('model') if isinstance(obj, dict) else obj
                model_feat = resolve_feature_cols_for_model(model, obj.get('features') if isinstance(obj, dict) else None, global_feat_cols)
                Xs = season_df.reindex(columns=model_feat).fillna(0.0)
                probs = predict_proba_safe(model, Xs, model_feat)
            ll = log_loss(y_true, probs)
            preds = probs.argmax(axis=1)
            acc = accuracy_score(y_true, preds)
            brier = multiclass_brier(y_true, probs)
            rows.append({'season': int(season), 'model': name, 'log_loss': float(ll), 'accuracy': float(acc), 'brier': float(brier)})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT, 'models_per_season_metrics.csv'), index=False)
    print('Saved per-season metrics to artifacts/models_per_season_metrics.csv')

    # Calibration curves per model and per class for test set
    n_bins = 10
    for name, m in models.items():
        model_feat = resolve_feature_cols_for_model(m, features_map.get(name), global_feat_cols)
        X_test = test.reindex(columns=model_feat).fillna(0.0)
        y_test = test['y'].values
        try:
            probs = predict_proba_safe(m, X_test, model_feat)
        except Exception:
            obj = joblib.load(MODEL_PATHS[name])
            model = obj.get('model') if isinstance(obj, dict) else obj
            model_feat = resolve_feature_cols_for_model(model, obj.get('features') if isinstance(obj, dict) else None, global_feat_cols)
            X_test = test.reindex(columns=model_feat).fillna(0.0)
            probs = predict_proba_safe(model, X_test, model_feat)

        plt.figure(figsize=(8, 6))
        for cls in range(probs.shape[1]):
            frac_pos, mean_pred = calibration_curve((y_test == cls).astype(int), probs[:, cls], n_bins=n_bins)
            plt.plot(mean_pred, frac_pos, marker='o', label=f'class {cls}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration curves (test) - {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f'calibration_{name}.png'))
        plt.close()

        # histogram of max predicted prob
        plt.figure(figsize=(6, 4))
        plt.hist(probs.max(axis=1), bins=20, alpha=0.7)
        plt.xlabel('Max predicted probability')
        plt.title(f'Prob histogram (test) - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f'prob_hist_{name}.png'))
        plt.close()

    print('Saved calibration curves and histograms to artifacts/')

    # Save a simple tuning summary (from last tuning run observed)
    tuning_summary = {
        'lightgbm_best': {'num_leaves': 31, 'n_estimators': 100, 'min_child_samples': 10, 'learning_rate': 0.01},
        'xgboost_best': {'subsample': 0.8, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01}
    }
    with open(TUNING_SUMMARY, 'w', encoding='utf-8') as fh:
        json.dump(tuning_summary, fh, indent=2)
    print('Saved tuning summary to', TUNING_SUMMARY)

    # Real prediction test: pick first match in 2024
    df2024 = df[df['Date'].dt.year == 2024]
    if df2024.empty:
        print('No 2024 matches found in features.csv')
        return
    match = df2024.iloc[0]
    preds = {}
    for name, m in models.items():
        try:
            model_feat = resolve_feature_cols_for_model(m, features_map.get(name), global_feat_cols)
            if isinstance(match, pd.Series):
                X_match = pd.DataFrame([match.to_dict()]).reindex(columns=model_feat).fillna(0.0)
            else:
                X_match = match.reindex(columns=model_feat).fillna(0.0)
            p = predict_proba_safe(m, X_match, model_feat)[0]
        except Exception:
            obj = joblib.load(MODEL_PATHS[name])
            model = obj.get('model') if isinstance(obj, dict) else obj
            model_feat = resolve_feature_cols_for_model(model, obj.get('features') if isinstance(obj, dict) else None, global_feat_cols)
            if isinstance(match, pd.Series):
                X_match = pd.DataFrame([match.to_dict()]).reindex(columns=model_feat).fillna(0.0)
            else:
                X_match = match.reindex(columns=model_feat).fillna(0.0)
            p = predict_proba_safe(model, X_match, model_feat)[0]
        preds[name] = {'probs': p.tolist(), 'pred': int(np.argmax(p))}

    out_pred = {
        'match_date': str(match['Date'].date()),
        'home': match.get('HomeTeam', '?'),
        'away': match.get('AwayTeam', '?'),
        'baseline': preds.get('baseline'),
        'lgb_calibrated': preds.get('lgb_calibrated'),
        'stacked': preds.get('stacked')
    }
    with open(os.path.join(OUT, 'single_match_prediction_2024.json'), 'w', encoding='utf-8') as fh:
        json.dump(out_pred, fh, indent=2)

    print('Saved single-match prediction to artifacts/single_match_prediction_2024.json')
    print('Prediction details:', out_pred)


if __name__ == '__main__':
    main()
