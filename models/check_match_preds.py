import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FEATS = os.path.join(ROOT, 'data', 'features.csv')
ART = os.path.join(ROOT, 'artifacts_regression')


def time_split(df):
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    return train, test


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    if 'Result' in df.columns:
        y_cls = df['Result'].map({'H':0,'D':1,'A':2})
    elif 'FTR' in df.columns:
        y_cls = df['FTR'].map({'H':0,'D':1,'A':2})
    else:
        raise RuntimeError('No Result/FTR')

    df['goal_diff'] = df.get('FTHG', 0) - df.get('FTAG', 0)
    exclude = {'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','Result','source_file'}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    train, test = time_split(df)
    X_test = test[feat_cols].fillna(0.0)
    y_test = y_cls.loc[test.index]

    artifacts_path = os.path.join(ART, 'match_regression_artifacts.joblib')
    if not os.path.exists(artifacts_path):
        print('Artifacts not found at', artifacts_path)
        return
    art = joblib.load(artifacts_path)
    reg = art.get('regressor')
    meta = art.get('meta')
    imp = art.get('imputer')
    sc = art.get('scaler')

    # ensure imputer/scaler available
    if imp is None:
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='median')
        imp.fit(train[feat_cols].fillna(0.0))
    if sc is None:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(imp.transform(train[feat_cols].fillna(0.0)))

    Xte_s = sc.transform(imp.transform(X_test))

    preds_reg = reg.predict(Xte_s)
    probs = meta.predict_proba(preds_reg.reshape(-1,1))

    sums = probs.sum(axis=1)
    sum_stats = {'min_sum': float(sums.min()), 'max_sum': float(sums.max()), 'mean_sum': float(sums.mean())}

    pred_labels = probs.argmax(axis=1)

    cm = confusion_matrix(y_test, pred_labels).tolist()
    acc = float(accuracy_score(y_test, pred_labels))
    ll = float(log_loss(y_test, probs, labels=[0,1,2]))

    per_class = {}
    for cls in [0,1,2]:
        idx = np.where(y_test.values == cls)[0]
        if len(idx) > 0:
            per_class[f'test_log_loss_class_{cls}'] = float(log_loss(y_test.values[idx], probs[idx], labels=[0,1,2]))
        else:
            per_class[f'test_log_loss_class_{cls}'] = None

    out = {'sum_stats': sum_stats, 'confusion_matrix': cm, 'accuracy': acc, 'log_loss': ll, 'per_class': per_class, 'n_test': int(len(y_test))}
    with open(os.path.join(ART, 'check_match_preds.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('Wrote', os.path.join(ART, 'check_match_preds.json'))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
