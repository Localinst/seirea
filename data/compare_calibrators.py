"""Compare Platt (logistic) vs Isotonic calibration for mapping predicted points -> P(winner).

Produces:
- artifacts_regression/calibration_compare_metrics.json
- artifacts_regression/calibration_compare.png

This script reuses the dataset preprocessing from the main training script but runs a
dedicated OOF routine to evaluate and compare calibrators.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_CSV = os.path.join(ROOT, 'dataset_cleaned_for_ml_with_pts_teamagg.csv')
ART = os.path.join(ROOT, 'artifacts_regression')
os.makedirs(ART, exist_ok=True)


def load_tpr_module():
    path = os.path.join(ROOT, 'data', 'train_points_regression.py')
    spec = importlib.util.spec_from_file_location('tpr', path)
    tpr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpr)
    return tpr


def main():
    tpr = load_tpr_module()
    df = tpr.load_and_clean(DATA_CSV)
    pre_df = tpr.create_preseason_features(df)
    pre_df = pre_df.dropna(subset=['rolling_points_3'], how='any')

    # select features same as main
    feature_cols = [c for c in pre_df.columns if c not in ['squad','season','season_year_end','points_target']]
    X = pre_df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    allnan = [c for c in X.columns if X[c].isna().all()]
    if allnan:
        X = X.drop(columns=allnan)
    zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if zero_var:
        X = X.drop(columns=zero_var)
    X_clean, _ = tpr.remove_highly_correlated(X, thresh=0.9)
    feature_cols = list(X_clean.columns)

    # build train subset
    train_df = pre_df[pre_df['season_year_end'] <= tpr.TRAIN_UP_TO_YEAR].copy()
    if 'winner' in train_df.columns:
        try:
            train_df['winner_flag'] = train_df['winner'].apply(lambda x: 1 if str(x).strip() in ['1','True','true'] else 0)
        except Exception:
            train_df['winner_flag'] = (train_df['winner'] == 1).astype(int)
    else:
        train_df['winner_flag'] = 0
        for s, g in train_df.groupby('season_year_end'):
            if g['points_target'].notna().any():
                idx = g['points_target'].idxmax()
                train_df.loc[idx, 'winner_flag'] = 1

    # OOF predictions using RF (+ XGB if available)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=tpr.RANDOM_STATE)
    oof = np.zeros(len(train_df))
    for tr_idx, va_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        Xtr = tr[feature_cols].copy()
        ytr = tr['points_target'].astype(float)
        Xva = va[feature_cols].copy()

        imp = SimpleImputer(strategy='median')
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
        Xva_s = sc.transform(imp.transform(Xva))

        rf = RandomForestRegressor(n_estimators=200, random_state=tpr.RANDOM_STATE, n_jobs=-1)
        rf.fit(Xtr_s, ytr)
        preds = rf.predict(Xva_s)

        if tpr.XGB_AVAILABLE:
            xgb = tpr.xgb.XGBRegressor(n_estimators=500, random_state=tpr.RANDOM_STATE)
            xgb.fit(Xtr_s, ytr)
            preds_x = xgb.predict(Xva_s)
            preds = 0.5 * preds + 0.5 * preds_x

        oof[va_idx] = preds

    y_win = train_df['winner_flag'].values

    # Fit Platt (Logistic) and Isotonic on OOF preds
    platt = LogisticRegression(solver='lbfgs', max_iter=2000)
    platt.fit(oof.reshape(-1,1), y_win)

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof, y_win)

    # Evaluate on OOF
    probs_platt = platt.predict_proba(oof.reshape(-1,1))[:,1]
    probs_iso = iso.predict(oof)

    metrics = {}
    for name, probs in [('platt', probs_platt), ('isotonic', probs_iso)]:
        metrics[name] = {
            'brier': float(brier_score_loss(y_win, probs)),
            'logloss': float(log_loss(y_win, probs)),
            'roc_auc': float(roc_auc_score(y_win, probs)) if len(np.unique(y_win))>1 else None,
            'acc_0.5': float(accuracy_score(y_win, (probs>=0.5).astype(int)))
        }

    # Calibration curves
    frac_pos_p, mean_pred_p = calibration_curve(y_win, probs_platt, n_bins=10)
    frac_pos_i, mean_pred_i = calibration_curve(y_win, probs_iso, n_bins=10)

    # Save metrics
    out_metrics = {
        'oof_count': int(len(oof)),
        'metrics': metrics
    }
    with open(os.path.join(ART, 'calibration_compare_metrics.json'), 'w') as f:
        json.dump(out_metrics, f, indent=2)

    # Plot comparison
    plt.figure(figsize=(6,6))
    plt.plot(mean_pred_p, frac_pos_p, 's-', label='Platt')
    plt.plot(mean_pred_i, frac_pos_i, 'o-', label='Isotonic')
    plt.plot([0,1],[0,1], 'k--', label='Perfect')
    plt.xlabel('Mean predicted prob')
    plt.ylabel('Fraction positive')
    plt.title('Calibration curve (OOF)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ART, 'calibration_compare.png'))
    plt.close()

    print('Saved calibration compare metrics and plot to', ART)


if __name__ == '__main__':
    main()
