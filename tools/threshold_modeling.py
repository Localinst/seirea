import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, mean_squared_error
import matplotlib.pyplot as plt

p_in = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\simulation_evaluation.csv")
out_report = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\threshold_model_report.csv")
out_roc = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\threshold_model_roc.png")
out_ev = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\threshold_model_ev_curve.png")

df = pd.read_csv(p_in, parse_dates=['Date'], dayfirst=True)
# Keep only rows with numeric odds
df = df.dropna(subset=['predicted_pct','fair_odds','max_bookie_odds'])

# Feature engineering
df['fair_odds'] = df['fair_odds'].astype(float)
df['max_bookie_odds'] = df['max_bookie_odds'].astype(float)
df['predicted_pct'] = df['predicted_pct'].astype(float)
df['bookie_gap'] = df['max_bookie_odds'] - df['fair_odds']
df['bookie_ratio'] = df['max_bookie_odds'] / df['fair_odds']
# predicted as categorical
if 'predicted' in df.columns:
    df['pred_is_home'] = (df['predicted'] == 'Home').astype(int)
    df['pred_is_draw'] = (df['predicted'] == 'Draw').astype(int)
    df['pred_is_away'] = (df['predicted'] == 'Away').astype(int)
else:
    df['pred_is_home'] = 0
    df['pred_is_draw'] = 0
    df['pred_is_away'] = 0

# Target for classification: correct (1/0)
df['correct_flag'] = df['correct'].astype(bool).astype(int)
# Target for regression EV per bet: profit if bet at max_bookie, else NaN
# profit_if_bet = (max_bookie_odds - 1) if correct else -1
stake = 1.0
df['profit_if_bet'] = df.apply(lambda r: (r['max_bookie_odds'] - 1.0)*stake if r['correct_flag']==1 else -stake, axis=1)

# Features list
features = ['predicted_pct','fair_odds','max_bookie_odds','bookie_gap','bookie_ratio','pred_is_home','pred_is_draw','pred_is_away']
X = df[features].fillna(0).values
y_clf = df['correct_flag'].values
y_reg = df['profit_if_bet'].values

# Classification: Logistic Regression with StratifiedKFold
clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print('Training logistic regression (cv)...')
probs = cross_val_predict(clf, X, y_clf, cv=cv, method='predict_proba')[:,1]
auc = roc_auc_score(y_clf, probs)
print(f'ROC AUC (cv): {auc:.3f}')

# Regression: GradientBoostingRegressor (cv predictions)
reg = GradientBoostingRegressor(random_state=42)
cvr = KFold(n_splits=5, shuffle=True, random_state=42)
print('Training regressor for profit (cv)...')
pred_ev = cross_val_predict(reg, X, y_reg, cv=cvr)
mse = mean_squared_error(y_reg, pred_ev)
print(f'Regression MSE (cv): {mse:.4f}')

# Save ROC plot
fpr, tpr, _ = roc_curve(y_clf, probs)
prec, rec, thr_pr = precision_recall_curve(y_clf, probs)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC')
plt.legend()

plt.subplot(1,2,2)
plt.plot(rec, prec)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall')
plt.tight_layout()
plt.savefig(out_roc, dpi=200)
print(f'Saved ROC/PR to {out_roc}')

# Add predictions back to df
df['pred_prob_cv'] = probs
df['pred_ev_cv'] = pred_ev

# Evaluate thresholds on predicted probability
thresholds = np.arange(0.3,0.81,0.01)
results = []
multiplier = 1.10  # keep same multiplier as base rule
for th in thresholds:
    capital = 100.0
    n_bets = 0
    wins = 0
    profits = []
    for _, r in df.iterrows():
        prob = r['pred_prob_cv']
        fair = r['fair_odds']
        maxb = r['max_bookie_odds']
        if (prob >= th) and (maxb >= fair * multiplier):
            n_bets += 1
            if r['correct_flag']==1:
                profit = (maxb - 1.0) * stake
                capital += profit
                wins += 1
                profits.append(profit)
            else:
                profit = -stake
                capital += profit
                profits.append(profit)
    final_cap = capital
    roi = (final_cap - 100.0) / 100.0 * 100
    winrate = (wins / n_bets * 100) if n_bets>0 else 0
    avg_profit = np.mean(profits) if len(profits)>0 else 0
    results.append({'threshold': th, 'n_bets': n_bets, 'wins': wins, 'winrate_pct': winrate, 'final_capital': final_cap, 'roi_pct': roi, 'avg_profit_per_bet': avg_profit})

res_df = pd.DataFrame(results)
res_df.to_csv(out_report, index=False)
print(f'Saved threshold report to {out_report}')

# Plot EV vs threshold
plt.figure(figsize=(8,4))
plt.plot(res_df['threshold'], res_df['final_capital'], marker='o')
plt.xlabel('Predicted probability threshold')
plt.ylabel('Final capital (EUR)')
plt.title('Final capital vs predicted probability threshold')
plt.grid(True)
plt.tight_layout()
plt.savefig(out_ev, dpi=200)
print(f'Saved EV curve to {out_ev}')

# Show top recommended thresholds
best_by_cap = res_df.sort_values('final_capital', ascending=False).iloc[:5]
print('Top thresholds by final capital:')
print(best_by_cap.to_string(index=False))

# Feature importance for regression using a single fit on full data
reg_full = GradientBoostingRegressor(random_state=42).fit(X, y_reg)
feat_imp = pd.DataFrame({'feature': features, 'importance': reg_full.feature_importances_}).sort_values('importance', ascending=False)
print('\nFeature importances (regression on profit):')
print(feat_imp.to_string(index=False))

# Save df with preds for inspection
out_preds = Path(r"C:\Users\ReadyToUse\Desktop\Data\serie a\artifacts\simulation_threshold_preds.csv")
df.to_csv(out_preds, index=False)
print(f'Saved per-match predictions to {out_preds}')
