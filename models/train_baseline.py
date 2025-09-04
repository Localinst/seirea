"""Train a baseline multinomial logistic regression on features.csv.
Saves model to artifacts/baseline_model.joblib and metrics to artifacts/metrics.csv
"""
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
OUT_MODEL = os.path.join(ROOT, 'artifacts', 'baseline_model.joblib')
OUT_METRICS = os.path.join(ROOT, 'artifacts', 'baseline_metrics.csv')


def prepare_Xy(df):
    # minimal features: recent_pts_mean, recent_gd_mean, recent_matches for home/away
    cols = ['home_recent_pts_mean','home_recent_gd_mean','home_recent_matches','away_recent_pts_mean','away_recent_gd_mean','away_recent_matches']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[cols].fillna(0.0)
    y = df['Result']
    return X, y


def time_split(df, date_col='Date', train_end='2023-12-31'):
    df = df.sort_values(date_col).copy()
    train = df[df[date_col] <= train_end]
    test = df[df[date_col] > train_end]
    return train, test


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    train, test = time_split(df)
    X_train, y_train = prepare_Xy(train)
    X_test, y_test = prepare_Xy(test)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(multi_class='multinomial', max_iter=1000))])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)
    preds = pipe.predict(X_test)
    ll = log_loss(y_test, probs, labels=pipe.classes_)
    acc = accuracy_score(y_test, preds)
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(pipe, OUT_MODEL)
    pd.DataFrame([{'log_loss': ll, 'accuracy': acc}]).to_csv(OUT_METRICS, index=False)
    print('Saved model to', OUT_MODEL)
    print('Metrics:', {'log_loss': ll, 'accuracy': acc})

if __name__ == '__main__':
    main()
