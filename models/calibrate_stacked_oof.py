"""Calibrate stacked model using time-aware OOF predictions.

This script expects an existing `artifacts/stacked_model.joblib` saved by `models/tune_and_stack.py`.
It will rebuild OOF stacked probabilities using TimeSeriesSplit, fit a multiclass logistic calibrator
on those OOF features, and save a calibrated wrapper to `artifacts/stacked_calibrated_model.joblib`.
It also writes calibration plots (before/after) to `artifacts/`.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
STACKED_IN = os.path.join(ROOT, 'artifacts', 'stacked_model.joblib')
STACKED_OUT = os.path.join(ROOT, 'artifacts', 'stacked_calibrated_model.joblib')
OUT_DIR = os.path.join(ROOT, 'artifacts')
os.makedirs(OUT_DIR, exist_ok=True)


class CalibratedStacked:
    """Top-level calibrated stacked wrapper that is pickleable."""
    def __init__(self, base_models, calibrator, feat_cols, classes_, base_names):
        self.base_models = base_models
        self.calibrator = calibrator
        self.feature_cols = feat_cols
        self.classes_ = classes_
        self._base_names = list(base_names)

    def predict_proba(self, X):
        Xsub = X.reindex(columns=self.feature_cols).fillna(0.0)
        probs = [self.base_models[name].predict_proba(Xsub) for name in self._base_names]
        stacked = np.hstack(probs)
        return self.calibrator.predict_proba(stacked)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)



def load_artifact(path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get('model'), obj.get('features')
    return obj, None


def main():
    # ensure StackedModel available for unpickling: import module and expose class under __main__
    try:
        import models.tune_and_stack as _ts
        try:
            import __main__ as _m
            setattr(_m, 'StackedModel', getattr(_ts, 'StackedModel'))
        except Exception:
            pass
    except Exception:
        fn = os.path.join(os.path.dirname(__file__), 'tune_and_stack.py')
        if os.path.exists(fn):
            spec = importlib.util.spec_from_file_location('models.tune_and_stack', fn)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['models.tune_and_stack'] = mod
            try:
                import __main__ as _m
                setattr(_m, 'StackedModel', getattr(mod, 'StackedModel'))
            except Exception:
                pass

    if not os.path.exists(STACKED_IN):
        print('stacked model artifact not found at', STACKED_IN)
        sys.exit(2)

    df = pd.read_csv(FEATS, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    y = df['Result'].map({'H':0,'D':1,'A':2})

    stacked_obj, feat_cols = load_artifact(STACKED_IN)
    if feat_cols is None:
        # try guess
        feat_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ('y',)]

    # Recreate base estimators list from stacked_obj
    # stacked_obj is expected to be StackedModel with .base_models dict and .meta_model
    base_models = getattr(stacked_obj, 'base_models', None)
    meta_model = getattr(stacked_obj, 'meta_model', None)
    classes_ = getattr(stacked_obj, 'classes_', None)
    if base_models is None or meta_model is None:
        print('stacked object does not expose base_models/meta_model; aborting')
        sys.exit(3)

    names = list(base_models.keys())

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)
    n = df.shape[0]
    n_classes = len(np.unique(y))
    n_base = len(base_models)
    oof = np.zeros((n, n_base * n_classes))
    filled = np.zeros(n, dtype=bool)

    names = list(base_models.keys())
    for i, name in enumerate(names):
        m = base_models[name]
        col_start = i * n_classes
        col_end = col_start + n_classes
        for train_idx, val_idx in tscv.split(df):
            Xtr = df.iloc[train_idx].reindex(columns=feat_cols).fillna(0.0)
            Xval = df.iloc[val_idx].reindex(columns=feat_cols).fillna(0.0)
            m_clone = clone(m)
            m_clone.fit(Xtr, y.iloc[train_idx])
            preds = m_clone.predict_proba(Xval)
            oof[val_idx, col_start:col_end] = preds
            filled[val_idx] = True
        # predict for missing earliest rows if any
        if not filled.all():
            missing = np.where(~filled)[0]
            preds = m.predict_proba(df.iloc[missing].reindex(columns=feat_cols).fillna(0.0))
            oof[missing, col_start:col_end] = preds

    # Fit multiclass logistic calibrator (Platt scaling) on OOF stacked probabilities
    calibrator = LogisticRegression(multi_class='multinomial', max_iter=2000)
    calibrator.fit(oof, y)

    # Evaluate before/after on held-out test portion (last fold)
    # We'll use last 20% as a quick holdout
    split_idx = int(n * 0.8)
    X_hold = df.iloc[split_idx:].reindex(columns=feat_cols).fillna(0.0)
    y_hold = y.iloc[split_idx:]
    # compute stacked probabilities (pre-calibration)
    probs_pre = stacked_obj.predict_proba(X_hold)
    # compute stacked oof features for hold
    # build hold stacked features by concatenating base model preds
    stacked_hold = []
    for name in names:
        preds = base_models[name].predict_proba(X_hold)
        stacked_hold.append(preds)
    stacked_hold = np.hstack(stacked_hold)
    probs_post = calibrator.predict_proba(stacked_hold)

    ll_pre = log_loss(y_hold, probs_pre)
    ll_post = log_loss(y_hold, probs_post)
    print('Hold log_loss pre-cal:', ll_pre, 'post-cal:', ll_post)

    # Save calibrated wrapper: keep base_models and calibrator; wrapper will mimic StackedModel but apply calibrator
    calibrated = CalibratedStacked(base_models, calibrator, feat_cols, classes_, names)
    joblib.dump({'model': calibrated, 'features': feat_cols}, STACKED_OUT)
    print('Saved calibrated stacked model to', STACKED_OUT)

    # plot calibration curves before/after for each class
    from sklearn.calibration import calibration_curve
    for cls in range(n_classes):
        frac_pre, mean_pre = calibration_curve((y_hold == cls).astype(int), probs_pre[:, cls], n_bins=10)
        frac_post, mean_post = calibration_curve((y_hold == cls).astype(int), probs_post[:, cls], n_bins=10)
        plt.figure(figsize=(6, 4))
        plt.plot(mean_pre, frac_pre, 'o-', label='pre-cal')
        plt.plot(mean_post, frac_post, 's-', label='post-cal')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.title(f'Calibration class {cls} (holdout)')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'calibration_stacked_class{cls}.png'))
        plt.close()

    # histogram of max predicted probability pre/post
    plt.figure(figsize=(6, 4))
    plt.hist(probs_pre.max(axis=1), bins=20, alpha=0.6, label='pre')
    plt.hist(probs_post.max(axis=1), bins=20, alpha=0.6, label='post')
    plt.legend()
    plt.title('Max predicted prob pre/post calibration (holdout)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'prob_hist_stacked_pre_post.png'))
    plt.close()


if __name__ == '__main__':
    main()
