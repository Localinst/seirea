"""Train LightGBM with TimeSeriesSplit and calibrate via CalibratedClassifierCV (sigmoid).
Saves calibrated model and metrics.
"""
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
OUT_DIR = os.path.join(ROOT, 'artifacts')
OUT_MODEL = os.path.join(OUT_DIR, 'lgb_calibrated.joblib')
OUT_METRICS = os.path.join(OUT_DIR, 'lgb_calibrated_metrics.csv')


def select_features(df):
    # prefer richer set if available
    prefer = [
        'home_recent_pts_mean','home_recent_gd_mean','home_recent_matches','home_days_since_last','home_days_since_log','home_elo',
        'away_recent_pts_mean','away_recent_gd_mean','away_recent_matches','away_days_since_last','away_days_since_log','away_elo',
        'elo_diff','h2h_home_goals_avg','h2h_away_goals_avg','h2h_home_pts_mean',
        'home_season_pts','away_season_pts','home_season_matches','away_season_matches',
        'recent_shots_sum','recent_goals_sum','recent_conversion','recent_pts_ma_3','recent_pts_ma_5','recent_pts_ma_10'
    ]
    cols = [c for c in prefer if c in df.columns]
    # fallback: numeric columns except some
    if len(cols)==0:
        exclude = {'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','Result','source_file'}
        cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def time_split(df):
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    return train, test


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    train, test = time_split(df)
    feat_cols = select_features(train)
    X_train = train[feat_cols].fillna(0.0)
    y_train = train['Result'].map({'H':0,'D':1,'A':2})
    X_test = test[feat_cols].fillna(0.0)
    y_test = test['Result'].map({'H':0,'D':1,'A':2})

    base = lgb.LGBMClassifier(objective='multiclass', num_class=3, learning_rate=0.05, num_leaves=31, n_estimators=200, random_state=42)
    # calibrate with TimeSeriesSplit CV inside CalibratedClassifierCV
    tscv = TimeSeriesSplit(n_splits=5)
    try:
        calib = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=tscv)
    except TypeError:
        calib = CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv=tscv)
    calib.fit(X_train, y_train)

    probs = calib.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    ll = log_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump({'model': calib, 'features': feat_cols}, OUT_MODEL)
    pd.DataFrame([{'log_loss': ll, 'accuracy': acc}]).to_csv(OUT_METRICS, index=False)
    print('Saved calibrated model to', OUT_MODEL)
    print('Metrics:', {'log_loss': ll, 'accuracy': acc})

if __name__ == '__main__':
    main()
