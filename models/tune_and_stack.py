"""Quick hyperparameter tuning (RandomizedSearch) for LGB and XGB, then stacking with logistic meta-learner.
Saves best estimators and stacked model + metrics.
"""
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
import lightgbm as lgb
try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
OUT_DIR = os.path.join(ROOT, 'artifacts')
OUT_STACK = os.path.join(OUT_DIR, 'stacked_model.joblib')
OUT_METR = os.path.join(OUT_DIR, 'stacked_metrics.csv')


class StackedModel:
    def __init__(self, base_models, meta_model, feature_cols, classes_):
        self.base_models = base_models  # dict name -> fitted estimator
        self.meta_model = meta_model
        self.feature_cols = feature_cols
        self.classes_ = classes_

    def predict_proba(self, X):
        # X: pd.DataFrame
        probs = []
        for name, m in self.base_models.items():
            p = m.predict_proba(X[self.feature_cols])
            probs.append(p)
        stacked = np.hstack(probs)
        return self.meta_model.predict_proba(stacked)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def train_stack_manual(X, y, estimators, meta, cv, feat_cols):
    # X: DataFrame, y: Series (encoded labels)
    n = X.shape[0]
    classes = np.unique(y)
    n_classes = len(classes)
    n_base = len(estimators)

    oof = np.zeros((n, n_base * n_classes))
    filled = np.zeros(n, dtype=bool)

    trained_full = {}

    # For each base estimator create OOF predictions
    for i, (name, est) in enumerate(estimators):
        col_start = i * n_classes
        col_end = col_start + n_classes
        # collect clones per fold
        for train_idx, val_idx in cv.split(X):
            est_clone = clone(est)
            est_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = est_clone.predict_proba(X.iloc[val_idx])
            oof[val_idx, col_start:col_end] = preds
            filled[val_idx] = True
        # retrain on full train for final use
        est_full = clone(est)
        est_full.fit(X, y)
        trained_full[name] = est_full

    # Fill any rows that were never in a validation fold (earliest rows)
    if not filled.all():
        missing_idx = np.where(~filled)[0]
        for i, (name, _) in enumerate(estimators):
            col_start = i * n_classes
            col_end = col_start + n_classes
            preds = trained_full[name].predict_proba(X.iloc[missing_idx])
            oof[missing_idx, col_start:col_end] = preds

    # Fit meta on OOF predictions
    meta.fit(oof, y)

    # Return a stacked model object with base models retrained on full data
    return StackedModel(trained_full, meta, feat_cols, classes)


def select_features(df):
    prefer = [
        'home_recent_pts_mean','home_recent_gd_mean','home_recent_matches','home_days_since_last','home_days_since_log','home_elo',
        'away_recent_pts_mean','away_recent_gd_mean','away_recent_matches','away_days_since_last','away_days_since_log','away_elo',
        'elo_diff','h2h_home_goals_avg','h2h_away_goals_avg','h2h_home_pts_mean',
        'home_season_pts','away_season_pts','home_season_matches','away_season_matches',
        'recent_shots_sum','recent_goals_sum','recent_conversion','recent_pts_ma_3','recent_pts_ma_5','recent_pts_ma_10'
    ]
    cols = [c for c in prefer if c in df.columns]
    if len(cols)==0:
        exclude = {'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','Result','source_file'}
        cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def time_split(df):
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    return train, test


def random_search(estimator, param_dist, X, y, cv, n_iter=20, n_jobs=1):
    rs = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring='neg_log_loss', n_jobs=n_jobs, random_state=42, refit=True)
    rs.fit(X, y)
    return rs


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    train, test = time_split(df)
    feat_cols = select_features(train)
    X_train = train[feat_cols].fillna(0.0)
    # target historically called 'Result'; fall back to 'FTR' if missing
    if 'Result' in train.columns:
        y_train = train['Result'].map({'H':0,'D':1,'A':2})
    elif 'FTR' in train.columns:
        y_train = train['FTR'].map({'H':0,'D':1,'A':2})
    else:
        raise RuntimeError('No target column found in features.csv: expected "Result" or "FTR"')

    X_test = test[feat_cols].fillna(0.0)
    if 'Result' in test.columns:
        y_test = test['Result'].map({'H':0,'D':1,'A':2})
    elif 'FTR' in test.columns:
        y_test = test['FTR'].map({'H':0,'D':1,'A':2})
    else:
        raise RuntimeError('No target column found in features.csv: expected "Result" or "FTR"')

    tscv = TimeSeriesSplit(n_splits=4)

    # LightGBM tuning
    lgb_base = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
    lgb_params = {
        'num_leaves': [15,31,63],
        'learning_rate': [0.01,0.03,0.05,0.1],
        'n_estimators': [100,200,400],
        'min_child_samples': [5,10,20]
    }
    print('Tuning LightGBM...')
    rs_lgb = random_search(lgb_base, lgb_params, X_train, y_train, cv=tscv, n_iter=12, n_jobs=1)
    best_lgb = rs_lgb.best_estimator_
    print('Best LGB params:', rs_lgb.best_params_)

    estimators = [('lgb', best_lgb)]

    # XGBoost tuning if available
    if has_xgb:
        xgb_base = XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        xgb_params = {
            'max_depth': [3,5,7],
            'learning_rate': [0.01,0.03,0.05,0.1],
            'n_estimators': [100,200,400],
            'subsample': [0.6,0.8,1.0]
        }
        print('Tuning XGBoost...')
        rs_xgb = random_search(xgb_base, xgb_params, X_train, y_train, cv=tscv, n_iter=12, n_jobs=1)
        best_xgb = rs_xgb.best_estimator_
        estimators.append(('xgb', best_xgb))
        print('Best XGB params:', rs_xgb.best_params_)
    else:
        print('XGBoost not available; skipping')

    # Train stacking using manual OOF generation compatible with TimeSeriesSplit
    print('Training stacking classifier (manual TimeSeriesSplit stacking)...')
    meta = LogisticRegression(max_iter=1000)
    stacked = train_stack_manual(X_train, y_train, estimators, meta, tscv, feat_cols)

    # Evaluate stack
    probs = stacked.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    ll = log_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump({'model': stacked, 'features': feat_cols}, OUT_STACK)
    pd.DataFrame([{'log_loss': ll, 'accuracy': acc}]).to_csv(OUT_METR, index=False)
    print('Saved stacked model to', OUT_STACK)
    print('Stack metrics:', {'log_loss': ll, 'accuracy': acc})

if __name__ == '__main__':
    main()
