import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score
from sklearn.calibration import calibration_curve

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FEATS = os.path.join(ROOT, 'data', 'features.csv')
ART = os.path.join(ROOT, 'artifacts_regression')
os.makedirs(ART, exist_ok=True)


def time_split(df):
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    return train, test


def load_artifacts():
    path = os.path.join(ART, 'match_regression_artifacts.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def ensure_meta_and_regressor(Xtr, ytr_reg, ytr_cls, artifacts):
    # returns (regressor, meta, imp, sc)
    if artifacts is not None and 'regressor' in artifacts and 'meta' in artifacts:
        return artifacts['regressor'], artifacts['meta'], artifacts.get('imputer'), artifacts.get('scaler')

    # else train simple versions
    imp = SimpleImputer(strategy='median')
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # OOF for mapping
    tscv = TimeSeriesSplit(n_splits=4)
    oof_preds = np.zeros(len(Xtr_s))
    for tr_idx, va_idx in tscv.split(Xtr_s):
        m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(Xtr_s[tr_idx], ytr_reg.iloc[tr_idx])
        oof_preds[va_idx] = m.predict(Xtr_s[va_idx])
    # train final
    rf.fit(Xtr_s, ytr_reg)
    meta = LogisticRegression(multi_class='multinomial', max_iter=2000)
    meta.fit(oof_preds.reshape(-1,1), ytr_cls.values)
    return rf, meta, imp, sc


def run():
    df = pd.read_csv(FEATS, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
        print('[ERR] need FTHG/FTAG in features.csv')
        return

    if 'Result' in df.columns:
        y_cls = df['Result'].map({'H':0,'D':1,'A':2})
    elif 'FTR' in df.columns:
        y_cls = df['FTR'].map({'H':0,'D':1,'A':2})
    else:
        print('[ERR] No Result/FTR column')
        return

    df['goal_diff'] = df['FTHG'] - df['FTAG']
    exclude = {'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','Result','source_file'}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    train, test = time_split(df)
    X_train = train[feat_cols].fillna(0.0)
    X_test = test[feat_cols].fillna(0.0)
    y_train_reg = train['goal_diff']
    y_test_reg = test['goal_diff']
    y_train_cls = y_cls.loc[train.index]
    y_test_cls = y_cls.loc[test.index]

    artifacts = load_artifacts()

    # if artifacts contains imputer/scaler, use them; else will be None and ensure will retrain
    reg, meta, imp, sc = ensure_meta_and_regressor(X_train, y_train_reg, y_train_cls, artifacts)

    # If imputer/scaler are None but artifacts existed, try to use artifacts keys
    if imp is None or sc is None:
        # if artifacts provided imputer/scaler earlier, load
        if artifacts is not None:
            imp = artifacts.get('imputer', SimpleImputer(strategy='median'))
            sc = artifacts.get('scaler', StandardScaler())
        else:
            imp = SimpleImputer(strategy='median')
            sc = StandardScaler()
            sc.fit(imp.fit_transform(X_train))

    Xtr_s = sc.transform(imp.transform(X_train)) if hasattr(sc, 'transform') else sc.fit_transform(imp.fit_transform(X_train))
    Xte_s = sc.transform(imp.transform(X_test)) if hasattr(sc, 'transform') else sc.transform(imp.transform(X_test))

    # compute OOF for regressor if not present
    tscv = TimeSeriesSplit(n_splits=4)
    oof_reg = np.zeros(len(Xtr_s))
    for tr_idx, va_idx in tscv.split(Xtr_s):
        m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(Xtr_s[tr_idx], y_train_reg.iloc[tr_idx])
        oof_reg[va_idx] = m.predict(Xtr_s[va_idx])

    # final reg predictions on test using provided reg if available
    try:
        preds_test_reg = reg.predict(Xte_s)
    except Exception:
        # fallback: train reg on full Xtr
        reg.fit(Xtr_s, y_train_reg)
        preds_test_reg = reg.predict(Xte_s)

    # meta probabilities
    probs_oof = meta.predict_proba(oof_reg.reshape(-1,1))
    probs_test = meta.predict_proba(preds_test_reg.reshape(-1,1))

    # sizes and class distributions
    stats = {
        'n_train': int(len(Xtr_s)),
        'n_test': int(len(Xte_s)),
        'test_class_counts': dict(pd.Series(y_test_cls).value_counts().sort_index().to_dict()),
        'train_class_counts': dict(pd.Series(y_train_cls).value_counts().sort_index().to_dict())
    }

    # confusion matrices
    cm_oof = confusion_matrix(y_train_cls, probs_oof.argmax(axis=1))
    cm_test = confusion_matrix(y_test_cls, probs_test.argmax(axis=1))

    # per-class log_loss (log_loss restricted to examples of each class)
    per_class = {}
    for cls in [0,1,2]:
        idx_tr = np.where(y_train_cls.values == cls)[0]
        idx_te = np.where(y_test_cls.values == cls)[0]
        if len(idx_tr) > 0:
            per_class[f'oof_log_loss_class_{cls}'] = float(log_loss(y_train_cls.values[idx_tr], probs_oof[idx_tr], labels=[0,1,2]))
        else:
            per_class[f'oof_log_loss_class_{cls}'] = None
        if len(idx_te) > 0:
            per_class[f'test_log_loss_class_{cls}'] = float(log_loss(y_test_cls.values[idx_te], probs_test[idx_te], labels=[0,1,2]))
        else:
            per_class[f'test_log_loss_class_{cls}'] = None

    # overall log_loss and accuracy
    metrics = {
        'oof_log_loss': float(log_loss(y_train_cls, probs_oof)),
        'test_log_loss': float(log_loss(y_test_cls, probs_test)),
        'oof_acc': float(accuracy_score(y_train_cls, probs_oof.argmax(axis=1))),
        'test_acc': float(accuracy_score(y_test_cls, probs_test.argmax(axis=1)))
    }

    out = {'stats': stats, 'metrics': metrics, 'per_class': per_class}
    joblib.dump(out, os.path.join(ART, 'match_regression_diagnostics.joblib'))
    import json
    with open(os.path.join(ART, 'match_regression_diagnostics.json'), 'w') as fh:
        json.dump(out, fh, indent=2)

    # save confusion matrix plots
    labels = ['Home','Draw','Away']
    plt.figure(figsize=(6,4))
    plt.imshow(cm_oof, cmap='Blues')
    plt.title('Confusion matrix (OOF)')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.xticks([0,1,2], labels)
    plt.yticks([0,1,2], labels)
    for i in range(cm_oof.shape[0]):
        for j in range(cm_oof.shape[1]):
            plt.text(j, i, str(cm_oof[i,j]), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(ART, 'confusion_oof.png'))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.imshow(cm_test, cmap='Blues')
    plt.title('Confusion matrix (Test)')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.xticks([0,1,2], labels)
    plt.yticks([0,1,2], labels)
    for i in range(cm_test.shape[0]):
        for j in range(cm_test.shape[1]):
            plt.text(j, i, str(cm_test[i,j]), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(ART, 'confusion_test.png'))
    plt.close()

    # calibration / reliability diagrams (one-vs-rest)
    plt.figure(figsize=(8,6))
    for cls in [0,1,2]:
        probs = probs_test[:, cls]
        true = (y_test_cls.values == cls).astype(int)
        frac_pos, mean_pred = calibration_curve(true, probs, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker='o', label=f'{labels[cls]}')
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction positives')
    plt.title('Reliability diagram (test)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ART, 'reliability_test.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    for cls in [0,1,2]:
        probs = probs_oof[:, cls]
        true = (y_train_cls.values == cls).astype(int)
        frac_pos, mean_pred = calibration_curve(true, probs, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker='o', label=f'{labels[cls]}')
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction positives')
    plt.title('Reliability diagram (OOF)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ART, 'reliability_oof.png'))
    plt.close()

    print('Diagnostics saved to', ART)


if __name__ == '__main__':
    run()
