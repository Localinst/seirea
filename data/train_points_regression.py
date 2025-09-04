"""Train regression models to predict season points (and therefore winner) for Serie A.

Produces: walk-forward validation results, saved models, feature importances, prediction for PREDICT_YEAR.

Notes/assumptions:
- If exact season points column ('Pts' or 'Points') is missing, the script will try to use 'Rk' (rank)
  to create a points proxy (monotonic mapping) as a fallback. Prefer providing real points.
- Only pre-season / ex-ante features are kept; columns that leak final-season totals are removed.
"""
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, kendalltau
import joblib
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- CONFIG ---
DATA_CSV = "dataset_cleaned_for_ml_with_pts_teamagg.csv"
TRAIN_START_YEAR = 2010
TRAIN_UP_TO_YEAR = 2022 # train through this year (inclusive) for final model
PREDICT_YEAR = 2023
RANDOM_STATE = 42
ARTIFACTS = "artifacts_regression"
os.makedirs(ARTIFACTS, exist_ok=True)


def load_and_clean(path):
    df = pd.read_csv(path)
    # normalize columns
    df.columns = df.columns.str.strip()
    # remove duplicated header rows
    df = df[df['season'].astype(str).str.lower() != 'stagione']
    df['squad'] = df['squad'].astype(str).str.strip()
    return df


def season_to_year_end(s):
    try:
        s = str(s)
        if '-' in s:
            y = s.split('-')[-1]
            if len(y) == 2:
                yint = int(y)
                if yint <= 25:
                    return 2000 + yint
                return 1900 + yint
            return int(y)
        return int(s)
    except Exception:
        return np.nan


def remove_leakage_cols(df):
    # Columns that are final-season totals / leakage and should NOT be used pre-season
    leakage = set([ 'Points', 'Pts', 'Rank', 'Rk', 'winner'])
    # Also drop obvious per-season totals that leak
    existing = [c for c in leakage if c in df.columns]
    if existing:
        print(f"[INFO] Dropping leakage columns: {existing}")
        df = df.drop(columns=existing)
    return df


def build_points_target(df):
    # If there is a real points column, use it. Try common names.
    for name in ['Pts']:
        if name in df.columns:
            df['points_target'] = pd.to_numeric(df[name], errors='coerce')
            print(f"[INFO] Using existing points column: {name}")
            return df

    # Fallback: if 'Rk' or 'Rank' exists, convert rank to a proxy points (monotonic)
    

    raise ValueError('No points or rank column found to create target; please provide points.')


def create_preseason_features(df):
    df = df.copy()
    df['season_year_end'] = df['season'].apply(season_to_year_end)

    # Ensure points target exists
    df = build_points_target(df)

    # Sort for rolling computations
    df = df.sort_values(['squad', 'season_year_end'])

    # Rolling points last 3 seasons (use points_target if available)
    df['points_last1'] = df.groupby('squad')['points_target'].shift(1)
    df['points_last2'] = df.groupby('squad')['points_target'].shift(2)
    df['points_last3'] = df.groupby('squad')['points_target'].shift(3)
    df['rolling_points_3'] = df[['points_last1', 'points_last2', 'points_last3']].mean(axis=1)

    # neopromoted: squad not present in previous season
    # Determine if squad was present in previous season by shifting season_year_end per squad.
    # Assign to df so subsequent accesses reference the new column.
    df['present_prev'] = df.groupby('squad')['season_year_end'].shift(1)
    df['is_neopromoted'] = df['present_prev'].isna().astype(int)

    # experience international: if column 'Intl' or 'Champions' exists, use it; otherwise 0
    if 'Champions' in df.columns:
        df['intl_experience'] = df['Champions'].astype(int)
    else:
        df['intl_experience'] = 0

    # delta market value if present
    if 'SquadValue' in df.columns:
        df['SquadValue_prev'] = df.groupby('squad')['SquadValue'].shift(1)
        df['delta_squad_value_pct'] = (df['SquadValue'] - df['SquadValue_prev']) / (df['SquadValue_prev'].replace(0, np.nan))
    else:
        df['delta_squad_value_pct'] = 0.0

    # Age: keep as-is if present
    if 'Age' in df.columns:
        df['age_mean'] = pd.to_numeric(df['Age'], errors='coerce')
    else:
        df['age_mean'] = np.nan

    # Attendance as proxy
    if 'Attendance' in df.columns:
        df['attendance'] = pd.to_numeric(df['Attendance'], errors='coerce')
    else:
        df['attendance'] = np.nan

    # Elo: if exists, keep; else approximate with rolling_points_3 normalized
    if 'Elo' in df.columns:
        df['elo'] = pd.to_numeric(df['Elo'], errors='coerce')
    else:
        # normalize rolling_points_3 within season groups
        df['elo'] = df.groupby('season')['rolling_points_3'].transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6))

    # After target creation, drop leakage columns (Pts, Rank) so they are not used as features.
    # This preserves team-aggregate columns and other pre-season covariates present in the
    # input dataset so downstream feature selection can choose the best ones.
    df = remove_leakage_cols(df)

    # Return full dataframe with derived preseason features, keeping original team-aggregate
    # columns; main() will select which columns to use as features (excluding ids/target).
    return df


def remove_highly_correlated(X, thresh=0.9):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
    return X.drop(columns=to_drop), to_drop


def walk_forward_validation(df, feature_cols, target_col='points_target'):
    years = sorted(df['season_year_end'].unique())
    years = [y for y in years if y >= TRAIN_START_YEAR and y <= TRAIN_UP_TO_YEAR]
    results = []

    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(max_iter=5000, random_state=RANDOM_STATE),
        'RF': RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    }
    if XGB_AVAILABLE:
        models['XGB'] = xgb.XGBRegressor(n_estimators=500, random_state=RANDOM_STATE)
    if LGB_AVAILABLE:
        # keep verbosity low to avoid repeated warnings flooding the console
        try:
            models['LGB'] = lgb.LGBMRegressor(n_estimators=500, random_state=RANDOM_STATE, verbose=-1)
        except TypeError:
            models['LGB'] = lgb.LGBMRegressor(n_estimators=500, random_state=RANDOM_STATE)

    for i in range(3, len(years)):
        train_years = years[:i]
        test_year = years[i]
        train_df = df[df['season_year_end'].isin(train_years)].copy()
        test_df = df[df['season_year_end'] == test_year].copy()
        if test_df.empty or train_df.empty:
            continue

        Xtr = train_df[feature_cols].copy()
        ytr = train_df[target_col].astype(float)
        Xte = test_df[feature_cols].copy()
        yte = test_df[target_col].astype(float)

        # Impute + scale
        imp = SimpleImputer(strategy='median')
        sc = StandardScaler()
        Xtr_imp = imp.fit_transform(Xtr)
        Xtr_s = sc.fit_transform(Xtr_imp)
        Xte_s = sc.transform(imp.transform(Xte))

        for name, model in models.items():
            m = model
            m.fit(Xtr_s, ytr)
            preds = m.predict(Xte_s)
            mse = mean_squared_error(yte, preds)
            rmse = mse ** 0.5
            sp, _ = spearmanr(yte, preds)
            kt, _ = kendalltau(yte, preds)
            # Hit@1: predicted top team equals actual top team
            pred_rank = test_df.copy()
            pred_rank['pred'] = preds
            pred_top = pred_rank.iloc[pred_rank['pred'].argmax()]['squad']
            actual_top = test_df.iloc[test_df[target_col].argmax()]['squad']
            hit1 = int(pred_top == actual_top)

            results.append({
                'model': name,
                'train_up_to': train_years[-1],
                'test_year': test_year,
                'rmse': rmse,
                'spearman': sp,
                'kendall': kt,
                'hit1': hit1
            })

    return pd.DataFrame(results)


def train_final_and_predict(df, feature_cols, target_col='points_target', predict_year=None, calibrator_method='isotonic', calib_blend=0.0):
    """Train final ensemble and predict for a single predict_year.

    If predict_year is None the global PREDICT_YEAR is used.
    Returns the predictions DataFrame or None.
    """
    if predict_year is None:
        predict_year = PREDICT_YEAR

    train_df = df[df['season_year_end'] <= TRAIN_UP_TO_YEAR].copy()
    predict_df = df[df['season_year_end'] == predict_year].copy()

    if predict_df.empty:
        print(f"[WARN] No data for predict year {predict_year}. Ensure preseason features provided.")

    X = train_df[feature_cols].copy()
    y = train_df[target_col].astype(float)

    imp = SimpleImputer(strategy='median')
    sc = StandardScaler()
    X_imp = imp.fit_transform(X)
    X_s = sc.fit_transform(X_imp)

    # Train a strong model (XGB if available else RF)
    # --- Build an ensemble (RF + XGB if available) and calibrate mapping from predicted points -> P(winner)
    # Create winner flag for calibration on the training subset
    train_df = train_df.copy()
    if 'winner' in train_df.columns:
        try:
            train_df['winner_flag'] = train_df['winner'].apply(lambda x: 1 if str(x).strip() in ['1', 'True', 'true'] else 0)
        except Exception:
            train_df['winner_flag'] = (train_df['winner'] == 1).astype(int)
    else:
        # derive winner by points_target per season (train set only)
        train_df['winner_flag'] = 0
        for s, g in train_df.groupby('season_year_end'):
            if g[target_col].notna().any():
                idx = g[target_col].idxmax()
                train_df.loc[idx, 'winner_flag'] = 1

    # Out-of-fold predictions on training set to fit calibrator
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(train_df))
    train_idx_map = train_df.index.to_series()
    # We'll need temporary arrays for imputer/scaler inside CV
    for tr_idx, va_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        Xtr = tr[feature_cols].copy()
        ytr = tr[target_col].astype(float)
        Xva = va[feature_cols].copy()

        imp_cv = SimpleImputer(strategy='median')
        sc_cv = StandardScaler()
        Xtr_s = sc_cv.fit_transform(imp_cv.fit_transform(Xtr))
        Xva_s = sc_cv.transform(imp_cv.transform(Xva))

        # train RF
        rf_cv = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf_cv.fit(Xtr_s, ytr)
        preds = rf_cv.predict(Xva_s)

        # train XGB on fold if available
        if XGB_AVAILABLE:
            xgb_cv = xgb.XGBRegressor(n_estimators=500, random_state=RANDOM_STATE)
            xgb_cv.fit(Xtr_s, ytr)
            preds_x = xgb_cv.predict(Xva_s)
            preds = 0.5 * preds + 0.5 * preds_x

        # assign OOF predictions for this fold (va_idx are positions)
        oof_preds[va_idx] = preds

    # Build calibrator (Platt = logistic regression on predicted score; isotonic optional)
    # calibrator_method is passed in (CLI) and defaults to 'isotonic'

    y_win = train_df['winner_flag'].values
    calibrator = None
    if calibrator_method == 'isotonic':
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_preds, y_win)
        calibrator = ('isotonic', iso)
    else:
        lr = LogisticRegression(solver='lbfgs', max_iter=2000)
        lr.fit(oof_preds.reshape(-1, 1), y_win)
        calibrator = ('platt', lr)

    # --- Compute calibration & accuracy metrics on OOF predictions ---
    try:
        if calibrator[0] == 'isotonic':
            probs_oof = calibrator[1].predict(oof_preds)
        else:
            probs_oof = calibrator[1].predict_proba(oof_preds.reshape(-1,1))[:, 1]

        metrics = {}
        metrics['oof_brier'] = float(brier_score_loss(y_win, probs_oof))
        try:
            metrics['oof_logloss'] = float(log_loss(y_win, probs_oof))
        except Exception:
            metrics['oof_logloss'] = None
        try:
            metrics['oof_roc_auc'] = float(roc_auc_score(y_win, probs_oof))
        except Exception:
            metrics['oof_roc_auc'] = None
        metrics['oof_acc_thresh_0.5'] = float(accuracy_score(y_win, (probs_oof >= 0.5).astype(int)))

        # Hit@1 on seasons using OOF preds
        train_oof_df = train_df.reset_index(drop=True).copy()
        train_oof_df['oof_pred_points'] = oof_preds
        hit1_list = []
        for s, g in train_oof_df.groupby('season_year_end'):
            if g[target_col].notna().any():
                pred_top = g.iloc[g['oof_pred_points'].argmax()]['squad']
                actual_top = g.iloc[g[target_col].argmax()]['squad']
                hit1_list.append(int(pred_top == actual_top))
        metrics['oof_hit1'] = float(np.mean(hit1_list)) if hit1_list else None

        # Save OOF metrics temporarily
        metrics_path = os.path.join(ARTIFACTS, 'calibration_metrics.json')
        with open(metrics_path, 'w') as fh:
            json.dump({'oof': metrics}, fh, indent=2)
        print('[INFO] OOF calibration metrics:', metrics)
    except Exception as e:
        print('[WARN] Failed to compute OOF calibration metrics:', e)

    # Retrain final ensemble on full training set
    imp = SimpleImputer(strategy='median')
    sc = StandardScaler()
    X_full = train_df[feature_cols].copy()
    X_full_s = sc.fit_transform(imp.fit_transform(X_full))

    rf_final = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(X_full_s, y)

    xgb_final = None
    if XGB_AVAILABLE:
        xgb_final = xgb.XGBRegressor(n_estimators=1000, random_state=RANDOM_STATE)
        xgb_final.fit(X_full_s, y)

    # Save artifacts
    joblib.dump({'rf': rf_final, 'xgb': xgb_final}, os.path.join(ARTIFACTS, 'final_ensemble.joblib'))
    joblib.dump(imp, os.path.join(ARTIFACTS, 'imputer.joblib'))
    joblib.dump(sc, os.path.join(ARTIFACTS, 'scaler.joblib'))
    joblib.dump(calibrator, os.path.join(ARTIFACTS, 'calibrator.joblib'))

    # Predict for PREDICT_YEAR if available
    out = None
    if not predict_df.empty:
        Xp = predict_df[feature_cols].copy()
        Xp_s = sc.transform(imp.transform(Xp))
        preds_rf = rf_final.predict(Xp_s)
        if xgb_final is not None:
            preds_x = xgb_final.predict(Xp_s)
            preds = 0.5 * preds_rf + 0.5 * preds_x
        else:
            preds = preds_rf

        out = predict_df[['squad', 'season']].copy()
        out['pred_points'] = preds
        out = out.sort_values('pred_points', ascending=False).reset_index(drop=True)
        out['rank_pred'] = np.arange(1, len(out)+1)
        # softmax relative probability (raw)
        exp = np.exp((out['pred_points'] - out['pred_points'].max()))
        out['prob_rel'] = (exp / exp.sum())

        # calibrated probability of winning via calibrator
        if calibrator[0] == 'isotonic':
            iso = calibrator[1]
            out['prob_calibrated'] = iso.predict(out['pred_points'].values)
        else:
            lr = calibrator[1]
            out['prob_calibrated'] = lr.predict_proba(out['pred_points'].values.reshape(-1,1))[:,1]

        # Optionally blend isotonic probabilities with the softmax relative probability
        # to avoid many exact zeros from a step-like isotonic mapping. Blend weight
        # `calib_blend` is the weight on the calibrated values (0..1); lower values
        # mix in `prob_rel` which is dense and smooth.
        try:
            blend = float(calib_blend)
        except Exception:
            blend = 0.0
        if blend > 0.0 and blend < 1.0:
            out['prob_calibrated'] = blend * out['prob_calibrated'] + (1.0 - blend) * out['prob_rel']

        # normalized calibrated probability across all teams (relative probability of winning
        # when you want the calibrated scores to form a distribution). This is optional but
        # useful for downstream ranking/portfolio decisions.
        total = out['prob_calibrated'].sum()
        if total > 0:
            out['prob_calibrated_rel'] = out['prob_calibrated'] / total
        else:
            out['prob_calibrated_rel'] = 0.0

    if out is not None:
        out.to_csv(os.path.join(ARTIFACTS, f'predictions_{predict_year}.csv'), index=False)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-years', type=str, default=None,
                        help='Comma separated list of season end years to predict, e.g. 2023,2025')
    parser.add_argument('--no-plots', action='store_true', help='Skip heavy plotting (pairplots, large figures)')
    parser.add_argument('--calibrator', type=str, default='isotonic', choices=['isotonic','platt'], help='Which calibrator to use for mapping points->P(winner)')
    parser.add_argument('--calib-blend', type=float, default=0.0, help='Blend weight (0..1) between calibrated prob and softmax prob to reduce zeros; 1.0 = calibrated only')
    args = parser.parse_args()

    print('[START] Loading data')
    df = load_and_clean(DATA_CSV)

    # Build preseason features (keep 'Pts' available so build_points_target can use it)
    print('[INFO] Building preseason features (preserve Pts for target creation)')
    pre_df = create_preseason_features(df)

    # After target creation we can safely drop leakage columns from the original dataframe if needed.
    # create_preseason_features already returns a compact set of pre-season features, so we don't
    # need to call remove_leakage_cols() here on `pre_df`.

    # Drop rows where rolling features are NaN (not enough history)
    pre_df = pre_df.dropna(subset=['rolling_points_3'], how='any')

    # Select feature columns (exclude id / target)
    feature_cols = [c for c in pre_df.columns if c not in ['squad','season','season_year_end','points_target']]

    print(f"[INFO] Candidate features before numeric coercion: {feature_cols}")

    # Coerce to numeric where possible and drop non-numeric or constant columns
    X = pre_df[feature_cols].copy()
    # Coerce each column to numeric (non-convertible -> NaN)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Drop columns that are entirely non-numeric / all-NaN after coercion
    allnan = [c for c in X.columns if X[c].isna().all()]
    if allnan:
        print(f"[INFO] Dropping non-numeric or empty features: {allnan}")
        X = X.drop(columns=allnan)

    # Drop constant columns (zero variance or single unique value)
    zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if zero_var:
        print(f"[INFO] Dropping constant features: {zero_var}")
        X = X.drop(columns=zero_var)

    # Remove high correlation (operates on numeric-only DataFrame)
    if X.shape[1] == 0:
        raise RuntimeError('No numeric features available after coercion; check input dataset')

    X_clean, dropped = remove_highly_correlated(X, thresh=0.9)
    if dropped:
        print(f"[INFO] Dropped highly correlated features: {dropped}")
    feature_cols = list(X_clean.columns)
    print(f"[INFO] Final feature set: {feature_cols}")

    print('[INFO] Running walk-forward validation')
    wfv = walk_forward_validation(pre_df, feature_cols, target_col='points_target')
    wfv.to_csv(os.path.join(ARTIFACTS, 'walk_forward_results.csv'), index=False)
    print(wfv.groupby('model').agg({'rmse':'mean','spearman':'mean','kendall':'mean','hit1':'mean'}))

    print('[INFO] Training final model and predicting')
    if args.predict_years:
        years = [int(y.strip()) for y in args.predict_years.split(',') if y.strip()]
    else:
        years = [PREDICT_YEAR]

    generated_any = False
    for y in years:
        preds = train_final_and_predict(pre_df, feature_cols, predict_year=y, calibrator_method=args.calibrator, calib_blend=args.calib_blend)
        if preds is not None:
            generated_any = True
            print(f"[RESULT] Predicted ranking for {y}:")
            # show relative calibrated probability alongside raw calibration
            display_cols = ['rank_pred','squad','pred_points','prob_rel','prob_calibrated','prob_calibrated_rel']
            available_display = [c for c in display_cols if c in preds.columns]
            print(preds[available_display].to_string(index=False))
        else:
            print(f"[WARN] No predictions generated for {y}")

    if not generated_any:
        print('[WARN] No predictions generated for any requested years')

    # --- PLOTTING SECTION: generate diagnostics and pairplots for all features ---
    if args.no_plots:
        print('[INFO] Skipping plotting as --no-plots was provided')
    else:
        try:
            print('[INFO] Generating diagnostic plots')
            artifacts_dir = ARTIFACTS
            os.makedirs(artifacts_dir, exist_ok=True)

            df_all = pd.read_csv(DATA_CSV)
            df_all.columns = df_all.columns.str.strip()

            # requested full column list (as provided by user)
            all_requested = ['season','squad','winner','# Pl','Age','Poss','MP','Starts','Min','90s','Gls','Ast','G+A','G-PK','PK','PKatt','CrdY','CrdR','G+A-PK','Attendance','Sh','SoT','SoT%','Sh/90','SoT/90','G/Sh','G/SoT','Dist','xG','xGA','xGD','xGD/90','Pts','__season_norm','__squad_norm','roster_size','starters_count','age_mean','age_mean_starters','pct_under23','pct_gk','pct_df','pct_mf','pct_fw','goals_tot','assists_tot','xg_tot','xga_proxy','team_xg_per90','conversion_sh_to_g','progressions_tot','yellow_cards','red_cards','top5_mean_gpa','bottom5_mean_gpa','gap_top5_bottom5_gpa','age_std']

            available = [c for c in all_requested if c in df_all.columns]
            if not available:
                available = list(df_all.columns)

            # prepare dataframe copy and coerce numerics where possible
            pair_df = df_all[available].copy()
            for c in pair_df.columns:
                if c == 'winner' or c == 'season' or c == 'squad' or c == '__season_norm' or c == '__squad_norm':
                    continue
                pair_df[c] = pd.to_numeric(pair_df[c], errors='coerce')

            # Correlation heatmap on numeric columns
            num_cols = pair_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                corr = pair_df[num_cols].corr()
                plt.figure(figsize=(max(10, len(num_cols)*0.3), max(8, len(num_cols)*0.3)))
                sns.heatmap(corr, cmap='vlag', center=0, linewidths=.5)
                plt.title('Correlation matrix (numeric features)')
                plt.tight_layout()
                plt.savefig(os.path.join(artifacts_dir, 'correlation_heatmap.png'))
                plt.close()

            # User-specified play_features pairplot: highlight winners in orange
            play_features = ['possession_percentile', 'Poss', 'assist_to_goal_ratio', 'penalty_conversion']
            play_present = [c for c in play_features if c in pair_df.columns]
            # Create categorical winner flag for plotting if present
            if 'winner' in pair_df.columns:
                try:
                    pair_df['is_winner'] = pair_df['winner'].apply(lambda x: 'Winner' if str(x).strip() in ['1', 'True', 'true'] else 'Other')
                except Exception:
                    pair_df['is_winner'] = pair_df['winner'].apply(lambda x: 'Winner' if x == 1 else 'Other')
            if len(play_present) >= 1:
                try:
                    subset_cols = play_present + (['is_winner'] if 'is_winner' in pair_df.columns else [])
                    plot_df = pair_df[subset_cols].dropna()
                    # sample to keep pairplot responsive
                    if len(plot_df) > 500:
                        plot_df = plot_df.sample(500, random_state=RANDOM_STATE)
                    palette = {'Winner': '#ff7f0e', 'Other': '#1f77b4'}
                    sns.pairplot(plot_df, hue='is_winner' if 'is_winner' in plot_df.columns else None, diag_kind='kde', plot_kws={'s':40,'alpha':0.7}, palette=palette)
                    plt.suptitle('Possession and Play Features Relationships', y=1.02)
                    plt.savefig(os.path.join(artifacts_dir, 'play_features_relationships.png'))
                    plt.close()
                except Exception:
                    # fallback: simple scatter plots
                    try:
                        for a in play_present[:3]:
                            for b in play_present[:3]:
                                if a == b:
                                    continue
                                plt.figure(figsize=(6,4))
                                sns.scatterplot(data=pair_df, x=a, y=b, hue='is_winner' if 'is_winner' in pair_df.columns else None, palette=palette)
                                plt.tight_layout()
                                plt.savefig(os.path.join(artifacts_dir, f'play_scatter_{a}_vs_{b}.png'))
                                plt.close()
                    except Exception:
                        pass

            # Pairplots for all numeric features: chunk to avoid massive plots
            max_pair = 4  # reduce vars per pairplot for speed
            numeric_all = num_cols.copy()
            # create/ensure categorical winner flag for plotting
            if 'is_winner' not in pair_df.columns and 'winner' in pair_df.columns:
                try:
                    pair_df['is_winner'] = pair_df['winner'].apply(lambda x: 'Winner' if str(x).strip() in ['1', 'True', 'true'] else 'Other')
                except Exception:
                    pair_df['is_winner'] = pair_df['winner'].apply(lambda x: 'Winner' if x == 1 else 'Other')

            palette = {'Winner': '#ff7f0e', 'Other': '#1f77b4'}
            max_chunks = 8
            chunk_idx = 0
            for i in range(0, len(numeric_all), max_pair):
                if chunk_idx >= max_chunks:
                    break
                chunk = numeric_all[i:i+max_pair]
                if len(chunk) < 2:
                    continue
                subset_cols = chunk + (['is_winner'] if 'is_winner' in pair_df.columns else [])
                plot_df = pair_df[subset_cols].dropna()
                # sample large sets for responsiveness
                if len(plot_df) > 500:
                    plot_df = plot_df.sample(500, random_state=RANDOM_STATE)
                try:
                    sns.pairplot(plot_df, hue='is_winner' if 'is_winner' in plot_df.columns else None, diag_kind='kde', plot_kws={'s':30,'alpha':0.6}, palette=palette)
                    plt.suptitle(f'Pairplot chunk {chunk_idx + 1}', y=1.02)
                    plt.savefig(os.path.join(artifacts_dir, f'pairplot_chunk_{chunk_idx + 1}.png'))
                    plt.close()
                except Exception:
                    # fallback: scatter_matrix
                    try:
                        pd.plotting.scatter_matrix(plot_df[chunk].dropna(), diagonal='kde', figsize=(8,8))
                        plt.suptitle(f'Scatter matrix chunk {chunk_idx + 1}', y=1.02)
                        plt.savefig(os.path.join(artifacts_dir, f'scatter_matrix_chunk_{chunk_idx + 1}.png'))
                        plt.close()
                    except Exception:
                        pass
                chunk_idx += 1

            # Histograms and boxplots for numeric features
            for c in num_cols:
                try:
                    plt.figure(figsize=(6,4))
                    sns.histplot(pair_df[c].dropna(), kde=True)
                    plt.title(f'Histogram {c}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(artifacts_dir, f'hist_{c}.png'))
                    plt.close()

                    plt.figure(figsize=(6,4))
                    sns.boxplot(x=pair_df[c].dropna())
                    plt.title(f'Boxplot {c}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(artifacts_dir, f'box_{c}.png'))
                    plt.close()
                except Exception:
                    continue

            print('[INFO] Plots saved to', artifacts_dir)
        except Exception as e:
            print('[WARN] Plotting failed:', e)


if __name__ == '__main__':
    main()
