"""Evaluate saved baseline model: confusion matrix and calibration plot.
Saves figures to artifacts/.
"""
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
MODEL = os.path.join(ROOT, 'artifacts', 'baseline_model.joblib')
OUT_DIR = os.path.join(ROOT, 'artifacts')


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    model = joblib.load(MODEL)
    # split as in train
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    X_test = test[['home_recent_pts_mean','home_recent_gd_mean','home_recent_matches','away_recent_pts_mean','away_recent_gd_mean','away_recent_matches']].fillna(0.0)
    y_test = test['Result']
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    # confusion
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    fig1, ax1 = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax1)
    fig1.suptitle('Confusion Matrix - Baseline')
    fig1.savefig(os.path.join(OUT_DIR, 'confusion_baseline.png'))
    plt.close(fig1)
    # calibration for each class
    for i, cls in enumerate(model.classes_):
        prob_pos = probs[:, i]
        y_bin = (y_test==cls).astype(int)
        frac_pos, mean_pred = calibration_curve(y_bin, prob_pos, n_bins=10)
        fig, ax = plt.subplots()
        ax.plot(mean_pred, frac_pos, 's-')
        ax.plot([0,1],[0,1],'--', color='gray')
        ax.set_xlabel('mean predicted prob')
        ax.set_ylabel('fraction of positives')
        ax.set_title(f'Calibration - class {cls}')
        fig.savefig(os.path.join(OUT_DIR, f'calibration_{cls}.png'))
        plt.close(fig)
    print('Saved evaluation figures to', OUT_DIR)

if __name__ == '__main__':
    main()
