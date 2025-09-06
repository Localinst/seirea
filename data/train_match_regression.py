"""Train a match-level regression baseline: predict goal difference then calibrate to outcome probabilities.

Saves OOF metrics and final test metrics to artifacts_regression/.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
OUT = os.path.join(ROOT, 'artifacts_regression')
os.makedirs(OUT, exist_ok=True)


def time_split(df):
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    return train, test


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # require score columns
    if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
        print('[ERR] features.csv missing FTHG/FTAG needed for match-level regression')
        return

    # target
    if 'Result' in df.columns:
        y_cls = df['Result'].map({'H':0,'D':1,'A':2})
    elif 'FTR' in df.columns:
        y_cls = df['FTR'].map({'H':0,'D':1,'A':2})
    else:
        print('[ERR] No Result/FTR column found')
        return

    df['goal_diff'] = df['FTHG'] - df['FTAG']

    exclude = {'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','Result','source_file'}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        print('[ERR] No numeric feature columns found')
        return

    train, test = time_split(df)
    X_train = train[feat_cols].fillna(0.0)
    X_test = test[feat_cols].fillna(0.0)
    y_train_reg = train['goal_diff']
    y_test_reg = test['goal_diff']
    y_train_cls = y_cls.loc[train.index]
    y_test_cls = y_cls.loc[test.index]

    # impute+scale
    imp = SimpleImputer(strategy='median')
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_train))
    Xte = sc.transform(imp.transform(X_test))

    # Use RF regressor as baseline
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=4)
    n = Xtr.shape[0]
    oof = np.zeros(n)

    for train_idx, val_idx in tscv.split(Xtr):
        rf_cv = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_cv.fit(Xtr[train_idx], y_train_reg.iloc[train_idx])
        preds = rf_cv.predict(Xtr[val_idx])
        oof[val_idx] = preds

    # fit final on full train
    rf.fit(Xtr, y_train_reg)
    preds_test_reg = rf.predict(Xte)

    # calibrate predicted goal_diff -> class probabilities via multinomial logistic on single feature
    meta = LogisticRegression(multi_class='multinomial', max_iter=2000)
    meta.fit(oof.reshape(-1,1), y_train_cls.values)

    probs_oof = meta.predict_proba(oof.reshape(-1,1))
    probs_test = meta.predict_proba(preds_test_reg.reshape(-1,1))

    metrics = {}
    try:
        metrics['oof_log_loss'] = float(log_loss(y_train_cls, probs_oof))
    except Exception:
        metrics['oof_log_loss'] = None
    try:
        metrics['test_log_loss'] = float(log_loss(y_test_cls, probs_test))
    except Exception:
        metrics['test_log_loss'] = None
    metrics['oof_acc'] = float(accuracy_score(y_train_cls, probs_oof.argmax(axis=1)))
    metrics['test_acc'] = float(accuracy_score(y_test_cls, probs_test.argmax(axis=1)))

    # save artifacts
    joblib.dump({'regressor': rf, 'imputer': imp, 'scaler': sc, 'meta': meta, 'features': feat_cols}, os.path.join(OUT, 'match_regression_artifacts.joblib'))
    pd.DataFrame([metrics]).to_json(os.path.join(OUT, 'match_regression_metrics.json'), orient='records')
    print('[INFO] Saved match regression metrics to', os.path.join(OUT, 'match_regression_metrics.json'))


if __name__ == '__main__':
    main()
