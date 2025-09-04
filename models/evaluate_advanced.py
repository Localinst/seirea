"""Evaluate LightGBM model: confusion, calibration, feature importance, and SHAP summary.
"""
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import shap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATS = os.path.join(ROOT, 'data', 'features.csv')
MODEL = os.path.join(ROOT, 'artifacts', 'lgb_model.joblib')
OUT_DIR = os.path.join(ROOT, 'artifacts')


def main():
    df = pd.read_csv(FEATS, parse_dates=['Date'])
    saved = joblib.load(MODEL)
    model = saved['model']
    feat_names = saved['features']
    # split
    train = df[df['Date'] <= '2023-12-31']
    test = df[df['Date'] > '2023-12-31']
    X_test = test[feat_names].fillna(0.0)
    y_test = test['Result']
    # get predicted probabilities and class preds
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)
    else:
        # fallback: model.predict may return labels
        preds_labels = model.predict(X_test)
        # produce dummy one-hot probs
        probs = np.zeros((len(preds_labels), 3))
        label_map = {'H':0,'D':1,'A':2}
        for i,l in enumerate(preds_labels):
            probs[i, label_map.get(l,0)] = 1.0
    preds = probs.argmax(axis=1)
    mapping = {0:'H',1:'D',2:'A'}
    preds_labels = [mapping[p] for p in preds]
    # confusion
    cm = confusion_matrix(y_test, preds_labels, labels=['H','D','A'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['H','D','A'])
    fig1, ax1 = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax1)
    fig1.suptitle('Confusion Matrix - LGB')
    fig1.savefig(os.path.join(OUT_DIR, 'confusion_lgb.png'))
    plt.close(fig1)
    # calibration per class
    for i, cls in enumerate(['H','D','A']):
        prob_pos = probs[:, i]
        y_bin = (y_test==cls).astype(int)
        # simple binned calibration
        bins = np.linspace(0,1,11)
        inds = np.digitize(prob_pos, bins) - 1
        frac = []
        meanp = []
        for b in range(len(bins)-1):
            sel = (inds==b)
            if sel.sum()>0:
                frac.append(y_bin[sel].mean())
                meanp.append(prob_pos[sel].mean())
        fig, ax = plt.subplots()
        if len(meanp)>0:
            ax.plot(meanp, frac, 's-')
        ax.plot([0,1],[0,1],'--', color='gray')
        ax.set_xlabel('mean predicted prob')
        ax.set_ylabel('fraction of positives')
        ax.set_title(f'Calibration - class {cls}')
        fig.savefig(os.path.join(OUT_DIR, f'calibration_lgb_{cls}.png'))
        plt.close(fig)
    # feature importance
    # retrieve importance robustly depending on model type
    try:
        imp_vals = model.feature_importances_
    except Exception:
        try:
            imp_vals = model.booster_.feature_importance(importance_type='gain')
        except Exception:
            imp_vals = [0]*len(feat_names)
    fi = pd.DataFrame({'feature': feat_names, 'importance': imp_vals})
    fi = fi.sort_values('importance', ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(fi['feature'][::-1], fi['importance'][::-1])
    ax.set_title('Feature importance (gain)')
    fig.savefig(os.path.join(OUT_DIR, 'feature_importance_lgb.png'))
    plt.close(fig)
    # SHAP summary (sample for speed)
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'shap_summary_lgb.png'))
    plt.close()
    print('Saved LGB evaluation artifacts to', OUT_DIR)

if __name__ == '__main__':
    main()
