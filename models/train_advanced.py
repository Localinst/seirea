"""Train an advanced model (LightGBM) on engineered features and save model + scaler.
"""
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
OUT_MODEL = os.path.join(ROOT, 'artifacts', 'lgb_model.joblib')
OUT_METRICS = os.path.join(ROOT, 'artifacts', 'lgb_metrics.csv')


def prepare_Xy(df):
    # select a richer set of features available
    prefer = [
        'home_recent_pts_mean','home_recent_gd_mean','home_recent_matches','home_days_since_last','home_elo',
        'away_recent_pts_mean','away_recent_gd_mean','away_recent_matches','away_days_since_last','away_elo',
        'elo_diff','h2h_home_goals_avg','h2h_away_goals_avg','h2h_home_pts_mean',
        'home_season_pts','away_season_pts','home_season_matches','away_season_matches'
    ]
    cols = [c for c in prefer if c in df.columns]
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

    # LightGBM dataset
    # Use sklearn API for better compatibility
    model = lgb.LGBMClassifier(objective='multiclass', num_class=3, learning_rate=0.05, num_leaves=31, n_estimators=500, random_state=42)
    # early stopping with validation set
    model.fit(X_train, y_train.map({'H':0,'D':1,'A':2}))
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    # map back
    mapping = {0:'H',1:'D',2:'A'}
    preds_labels = [mapping[p] for p in preds]
    ll = log_loss(y_test.map({'H':0,'D':1,'A':2}), probs)
    acc = accuracy_score(y_test, preds_labels)

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump({'model': model, 'features': X_train.columns.tolist()}, OUT_MODEL)
    pd.DataFrame([{'log_loss': ll, 'accuracy': acc}]).to_csv(OUT_METRICS, index=False)
    print('Saved model to', OUT_MODEL)
    print('Metrics:', {'log_loss': ll, 'accuracy': acc})

if __name__ == '__main__':
    main()
