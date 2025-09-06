"""Detailed evaluation suite

Performs:
 A) Cross-season CV/train/test metrics for each base learner and stacked
 B) Learning curves (increasing train size) for one base and the meta-model
 C) Feature importances and permutation importance
 D) Runs regression->calibration script `data/train_points_regression.py` and records its outputs

Saves metrics and plots under `artifacts/` and `artifacts_regression/`.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import subprocess
import json

ROOT = os.path.dirname(os.path.dirname(__file__))
# Ensure project root is on sys.path so `import models` works when this file is executed from models/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FEATS = os.path.join(ROOT, 'data', 'features.csv')
STACKED = os.path.join(ROOT, 'artifacts', 'stacked_model.joblib')
OUT = os.path.join(ROOT, 'artifacts')
OUT_REG = os.path.join(ROOT, 'artifacts_regression')
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT_REG, exist_ok=True)


def time_split(df):
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    return train, test


def multiclass_brier(y_true, probs):
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[y_true]
    return np.mean((probs - onehot) ** 2)


def task_A_cross_season():
    print('Running A: cross-season CV/train/test metrics')
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Robust load: joblib pickles sometimes reference classes under __main__ when saved from scripts.
    # Try normal load, otherwise import likely defining modules and inject classes into this module, then retry.
    # Preload likely class definitions into __main__ so joblib can unpickle objects that were
    # saved from scripts where classes lived in __main__ at save time.
    try:
        import models.tune_and_stack as mts
        sys.modules['__main__'].StackedModel = getattr(mts, 'StackedModel', None)
    except Exception:
        pass
    try:
        import models.calibrated_stack_wrapper as mcw
        sys.modules['__main__'].CalibratedStacked = getattr(mcw, 'CalibratedStacked', None)
    except Exception:
        pass

    obj = joblib.load(STACKED)
    stacked = obj.get('model') if isinstance(obj, dict) else obj
    feat_cols = obj.get('features') if isinstance(obj, dict) else None
    if feat_cols is None:
        feat_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ('y','Result')]

    train, test = time_split(df)
    X_train = train.reindex(columns=feat_cols).fillna(0.0)
    X_test = test.reindex(columns=feat_cols).fillna(0.0)
    # target
    if 'Result' in train.columns:
        y_train = train['Result'].map({'H':0,'D':1,'A':2})
        y_test = test['Result'].map({'H':0,'D':1,'A':2})
    else:
        y_train = train['FTR'].map({'H':0,'D':1,'A':2})
        y_test = test['FTR'].map({'H':0,'D':1,'A':2})

    # base models
    base_models = getattr(stacked, 'base_models', {})
    names = list(base_models.keys())

    tscv = TimeSeriesSplit(n_splits=4)
    n = X_train.shape[0]
    classes = np.unique(y_train)
    n_classes = len(classes)

    results = []

    # For each base model compute CV OOF, train and test metrics
    for name in names:
        print('Evaluating base:', name)
        est = base_models[name]
        # OOF
        oof = np.zeros((n, n_classes))
        filled = np.zeros(n, dtype=bool)
        for train_idx, val_idx in tscv.split(X_train):
            m = clone(est)
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = m.predict_proba(X_train.iloc[val_idx])
            oof[val_idx] = preds
            filled[val_idx] = True
        # fill missing earliest rows
        if not filled.all():
            missing = np.where(~filled)[0]
            m_full = clone(est)
            m_full.fit(X_train, y_train)
            oof[missing] = m_full.predict_proba(X_train.iloc[missing])

        # fit final on full train (for test eval)
        final = clone(est)
        final.fit(X_train, y_train)

        # metrics
        try:
            ll_cv = log_loss(y_train, oof)
        except Exception:
            ll_cv = float('nan')
        preds_train = final.predict_proba(X_train)
        preds_test = final.predict_proba(X_test)
        ll_train = log_loss(y_train, preds_train)
        ll_test = log_loss(y_test, preds_test)
        acc_train = accuracy_score(y_train, preds_train.argmax(axis=1))
        acc_cv = accuracy_score(y_train, oof.argmax(axis=1))
        acc_test = accuracy_score(y_test, preds_test.argmax(axis=1))

        results.append({'model': name, 'phase': 'train', 'log_loss': ll_train, 'accuracy': acc_train})
        results.append({'model': name, 'phase': 'cv', 'log_loss': ll_cv, 'accuracy': acc_cv})
        results.append({'model': name, 'phase': 'test', 'log_loss': ll_test, 'accuracy': acc_test})

    # For stacked: rebuild meta via OOF from base CV folds
    print('Evaluating stacked (rebuilding meta on OOF)')
    # Build OOF matrix for bases
    n_base = len(names)
    oof_stack = np.zeros((n, n_base * n_classes))
    filled = np.zeros(n, dtype=bool)
    for i, name in enumerate(names):
        est = base_models[name]
        col_start = i * n_classes
        col_end = col_start + n_classes
        for train_idx, val_idx in tscv.split(X_train):
            m = clone(est)
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = m.predict_proba(X_train.iloc[val_idx])
            oof_stack[val_idx, col_start:col_end] = preds
            filled[val_idx] = True
        if not filled.all():
            missing = np.where(~filled)[0]
            preds = base_models[name].predict_proba(X_train.iloc[missing])
            oof_stack[missing, col_start:col_end] = preds

    # train meta on OOF
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(max_iter=2000)
    meta.fit(oof_stack, y_train)

    # stacked pred on test: use base final models trained on full train
    stacked_feats_test = []
    for name in names:
        m = base_models[name]
        stacked_feats_test.append(m.predict_proba(X_test))
    stacked_test = np.hstack(stacked_feats_test)
    probs_test = meta.predict_proba(stacked_test)
    ll_stacked_test = log_loss(y_test, probs_test)
    # for cv, compute meta preds on holdout rows from OOF by training meta already on oof_stack
    probs_cv_meta = meta.predict_proba(oof_stack)
    ll_stacked_cv = log_loss(y_train, probs_cv_meta)
    preds_test = probs_test.argmax(axis=1)
    acc_test = accuracy_score(y_test, preds_test)

    results.append({'model': 'stacked', 'phase': 'cv', 'log_loss': ll_stacked_cv, 'accuracy': float('nan')})
    results.append({'model': 'stacked', 'phase': 'test', 'log_loss': ll_stacked_test, 'accuracy': acc_test})

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT, 'cross_season_metrics_per_model.csv'), index=False)
    print('Saved cross-season metrics to', os.path.join(OUT, 'cross_season_metrics_per_model.csv'))


def task_B_learning_curves():
    print('Running B: learning curves')
    df = pd.read_csv(FEATS, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    obj = joblib.load(STACKED)
    stacked = obj.get('model') if isinstance(obj, dict) else obj
    feat_cols = obj.get('features') if isinstance(obj, dict) else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    train, test = time_split(df)
    X_train = train.reindex(columns=feat_cols).fillna(0.0)
    if 'Result' in train.columns:
        y_train = train['Result'].map({'H':0,'D':1,'A':2})
    else:
        y_train = train['FTR'].map({'H':0,'D':1,'A':2})

    # choose base and meta
    base_name = list(stacked.base_models.keys())[0]
    base_est = stacked.base_models[base_name]
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(max_iter=2000)

    sizes = np.linspace(0.2, 1.0, 6)
    train_scores_base = []
    val_scores_base = []
    train_scores_meta = []
    val_scores_meta = []

    # fixed validation window: last 20% of training
    split_idx = int(len(X_train) * 0.8)
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]

    for frac in sizes:
        nrows = max(50, int(len(X_train) * frac))
        X_sub = X_train.iloc[:nrows]
        y_sub = y_train.iloc[:nrows]

        # base
        b = clone(base_est)
        b.fit(X_sub, y_sub)
        p_train = b.predict_proba(X_sub)
        p_val = b.predict_proba(X_val)
        train_scores_base.append(log_loss(y_sub, p_train))
        val_scores_base.append(log_loss(y_val, p_val))

        # meta: build simple stacked features using base trained on X_sub and using X_val for val
        # We will use the single base repeated to form a minimal stacked meta input
        stacked_train = b.predict_proba(X_sub)
        stacked_val = b.predict_proba(X_val)
        mm = clone(meta)
        try:
            mm.fit(stacked_train, y_sub)
            train_scores_meta.append(log_loss(y_sub, mm.predict_proba(stacked_train)))
            val_scores_meta.append(log_loss(y_val, mm.predict_proba(stacked_val)))
        except Exception:
            train_scores_meta.append(float('nan'))
            val_scores_meta.append(float('nan'))

    # plot
    plt.figure()
    plt.plot(sizes, train_scores_base, '-o', label='base train')
    plt.plot(sizes, val_scores_base, '-o', label='base val')
    plt.plot(sizes, train_scores_meta, '-s', label='meta train')
    plt.plot(sizes, val_scores_meta, '-s', label='meta val')
    plt.xlabel('Fraction of training data')
    plt.ylabel('log loss')
    plt.legend()
    plt.title('Learning curves (base & meta)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'learning_curves_base_meta.png'))
    plt.close()
    print('Saved learning curves to', os.path.join(OUT, 'learning_curves_base_meta.png'))


def task_C_feature_importance():
    print('Running C: feature importances and permutation')
    df = pd.read_csv(FEATS, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    obj = joblib.load(STACKED)
    stacked = obj.get('model') if isinstance(obj, dict) else obj
    feat_cols = obj.get('features') if isinstance(obj, dict) else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    train, test = time_split(df)
    X_train = train.reindex(columns=feat_cols).fillna(0.0)
    X_test = test.reindex(columns=feat_cols).fillna(0.0)
    if 'Result' in train.columns:
        y_train = train['Result'].map({'H':0,'D':1,'A':2})
        y_test = test['Result'].map({'H':0,'D':1,'A':2})
    else:
        y_train = train['FTR'].map({'H':0,'D':1,'A':2})
        y_test = test['FTR'].map({'H':0,'D':1,'A':2})

    # pick LGB base if present
    base_name = None
    for n in stacked.base_models:
        if n.lower().startswith('lgb') or 'lgb' in n.lower():
            base_name = n
            break
    if base_name is None:
        base_name = list(stacked.base_models.keys())[0]

    base = stacked.base_models[base_name]
    try:
        importances = None
        if hasattr(base, 'feature_importances_'):
            importances = base.feature_importances_
            fi = pd.DataFrame({'feature': feat_cols, 'importance': importances})
            fi = fi.sort_values('importance', ascending=False)
            fi.to_csv(os.path.join(OUT, f'feature_importances_{base_name}.csv'), index=False)
    except Exception:
        pass

    # permutation importance for base on test set
    try:
        print('Computing permutation importance for', base_name)
        r = permutation_importance(base, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
        perm = pd.DataFrame({'feature': feat_cols, 'importance_mean': r.importances_mean, 'importance_std': r.importances_std})
        perm = perm.sort_values('importance_mean', ascending=False)
        perm.to_csv(os.path.join(OUT, f'perm_importance_{base_name}.csv'), index=False)
    except Exception as e:
        print('Permutation importance failed:', e)

    print('Saved feature importances to', OUT)


def task_D_run_regression():
    print('Running D: training regression-based pipeline')
    # run data/train_points_regression.py as a separate process
    # prefer match-level regression baseline (predict goal diff -> calibrate to match outcome)
    script = os.path.join(ROOT, 'data', 'train_match_regression.py')
    if not os.path.exists(script):
        print('Regression training script not found at', script)
        return
    try:
        # run in a subprocess and capture output
        proc = subprocess.run([os.sys.executable, script], cwd=ROOT, capture_output=True, text=True, timeout=1200)
        with open(os.path.join(OUT_REG, 'train_points_regression_stdout.txt'), 'w', encoding='utf-8') as fh:
            fh.write(proc.stdout + '\n--- ERR ---\n' + proc.stderr)
        print('Ran regression training; output saved to', os.path.join(OUT_REG, 'train_points_regression_stdout.txt'))
    except Exception as e:
        print('Error running regression script:', e)


def main():
    task_A_cross_season()
    task_B_learning_curves()
    task_C_feature_importance()
    task_D_run_regression()


if __name__ == '__main__':
    main()
