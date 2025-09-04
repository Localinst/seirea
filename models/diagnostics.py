"""Compute per-season diagnostics comparing baseline and LightGBM models.
Saves metrics CSV and plots to artifacts/.
"""
import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
BASELINE = os.path.join(ROOT, 'artifacts', 'baseline_model.joblib')
LGB = os.path.join(ROOT, 'artifacts', 'lgb_model.joblib')
OUT_DIR = os.path.join(ROOT, 'artifacts')
OUT_CSV = os.path.join(OUT_DIR, 'diagnostics_per_season.csv')


def extract_season(srcfile):
    try:
        base = os.path.basename(srcfile)
        if base.startswith('season-'):
            return base.split('-')[1].split('.')[0]
    except Exception:
        pass
    return 'unknown'


def safe_predict_proba(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    # try sklearn pipeline
    try:
        return model.predict_proba(X)
    except Exception:
        # fallback to predict -> one-hot
        preds = model.predict(X)
        probs = np.zeros((len(preds), 3))
        map_lbl = {'H':0,'D':1,'A':2}
        for i,l in enumerate(preds):
            probs[i, map_lbl.get(l,0)] = 1.0
        return probs


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    # load models
    baseline = joblib.load(BASELINE)
    lgb_saved = joblib.load(LGB)
    lgb_model = lgb_saved['model']
    lgb_feats = lgb_saved['features']

    rows = []
    seasons = sorted(df['source_file'].dropna().unique())
    for src in seasons:
        season = extract_season(src)
        sub = df[df['source_file']==src]
        if sub.empty:
            continue
        y = sub['Result']
        # baseline features
        base_cols = ['home_recent_pts_mean','home_recent_gd_mean','home_recent_matches','away_recent_pts_mean','away_recent_gd_mean','away_recent_matches']
        X_base = sub[[c for c in base_cols if c in sub.columns]].fillna(0.0)
        probs_base = safe_predict_proba(baseline, X_base)
        # lgb features
        X_lgb = sub[[c for c in lgb_feats if c in sub.columns]].fillna(0.0)
        probs_lgb = safe_predict_proba(lgb_model, X_lgb)
        # map y to numeric for log_loss for lgb if needed
        map_num = {'H':0,'D':1,'A':2}
        y_num = y.map(map_num)
        # compute metrics
        try:
            ll_base = log_loss(y, probs_base)
        except Exception:
            # if baseline pipeline returns classes in different order, try align
            ll_base = log_loss(y_num, probs_base)
        ll_lgb = log_loss(y_num, probs_lgb)
        acc_base = accuracy_score(y, probs_base.argmax(axis=1).astype(object).astype(str)) if probs_base.ndim>1 else accuracy_score(y, (probs_base.argmax(axis=1)))
        acc_lgb = accuracy_score(y, probs_lgb.argmax(axis=1).astype(object).astype(str)) if probs_lgb.ndim>1 else accuracy_score(y, (probs_lgb.argmax(axis=1)))
        # brier for multiclass: average of class-wise
        brier_base = np.mean([brier_score_loss((y==lbl).astype(int), probs_base[:,i]) for i,lbl in enumerate(['H','D','A'])])
        brier_lgb = np.mean([brier_score_loss((y==lbl).astype(int), probs_lgb[:,i]) for i,lbl in enumerate(['H','D','A'])])

        rows.append({'source_file': src, 'season': season, 'n_matches': len(sub), 'logloss_base': ll_base, 'logloss_lgb': ll_lgb, 'acc_base': acc_base, 'acc_lgb': acc_lgb, 'brier_base': brier_base, 'brier_lgb': brier_lgb})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # plots: logloss per season
    if not out.empty:
        x = out['season']
        plt.figure(figsize=(10,5))
        plt.plot(x, out['logloss_base'], marker='o', label='baseline')
        plt.plot(x, out['logloss_lgb'], marker='o', label='lgb')
        plt.xticks(rotation=45)
        plt.ylabel('log loss')
        plt.title('Log loss per season')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'logloss_per_season.png'))
        plt.close()

        # probability histograms
        for name, probs in [('baseline', probs_base), ('lgb', probs_lgb)]:
            plt.figure(figsize=(8,4))
            plt.hist(probs.flatten(), bins=50)
            plt.title(f'Predicted probability distribution ({name})')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'probs_hist_{name}.png'))
            plt.close()

    print('Saved diagnostics to', OUT_DIR)

if __name__ == '__main__':
    main()
