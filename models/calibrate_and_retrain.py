"""Calibrate LightGBM model with Platt (sigmoid) and Isotonic using CalibratedClassifierCV.
Saves calibrated models and metrics.
"""
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
LGB = os.path.join(ROOT, 'artifacts', 'lgb_model.joblib')
OUT_DIR = os.path.join(ROOT, 'artifacts')
OUT_METRICS = os.path.join(OUT_DIR, 'calibrated_metrics.csv')


def load_model():
    saved = joblib.load(LGB)
    return saved['model'], saved['features']


def prepare_Xy(df, feat_names):
    X = df[feat_names].fillna(0.0)
    y = df['Result'].map({'H':0,'D':1,'A':2})
    return X, y


def time_split(df):
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    return train, test


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    model, feat_names = load_model()
    train, test = time_split(df)
    X_train, y_train = prepare_Xy(train, feat_names)
    X_test, y_test = prepare_Xy(test, feat_names)

    # Calibrate with sigmoid (Platt)
    try:
        calibrator_sig = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
    except TypeError:
        calibrator_sig = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
    calibrator_sig.fit(X_test, y_test)
    # Isotonic (may require more data)
    try:
        try:
            calibrator_iso = CalibratedClassifierCV(estimator=model, method='isotonic', cv='prefit')
        except TypeError:
            calibrator_iso = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv='prefit')
        calibrator_iso.fit(X_test, y_test)
    except Exception as e:
        calibrator_iso = None
        print('Isotonic calibration failed:', e)

    # evaluate
    def eval_model(m, X, y):
        probs = m.predict_proba(X)
        ll = log_loss(y, probs)
        acc = accuracy_score(y, probs.argmax(axis=1))
        brier = np.mean([brier_score_loss((y==i).astype(int), probs[:,i]) for i in range(probs.shape[1])])
        return ll, acc, brier

    results = []
    # raw model
    ll_raw, acc_raw, brier_raw = eval_model(model, X_test, y_test)
    results.append({'model':'raw_lgb','log_loss': ll_raw, 'accuracy': acc_raw, 'brier': brier_raw})
    # sigmoid
    ll_sig, acc_sig, brier_sig = eval_model(calibrator_sig, X_test, y_test)
    results.append({'model':'cal_sigmoid','log_loss': ll_sig, 'accuracy': acc_sig, 'brier': brier_sig})
    # isotonic
    if calibrator_iso is not None:
        ll_iso, acc_iso, brier_iso = eval_model(calibrator_iso, X_test, y_test)
        results.append({'model':'cal_isotonic','log_loss': ll_iso, 'accuracy': acc_iso, 'brier': brier_iso})

    out = pd.DataFrame(results)
    out.to_csv(OUT_METRICS, index=False)
    # save calibrators
    joblib.dump({'sigmoid': calibrator_sig, 'isotonic': calibrator_iso}, os.path.join(OUT_DIR, 'calibrators.joblib'))
    print('Saved calibration results and models to', OUT_DIR)

if __name__ == '__main__':
    main()
