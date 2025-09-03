# train_xg_winner.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
DATA_CSV = "dataset_cleaned_for_ml.csv"  # file che contiene tutte le righe team-season, con colonna 'season' e 'squad'
TRAIN_UP_TO_YEAR = 2023 # includi stagione 2022 come training
PREDICT_YEAR = 2024  # predici per 2023
RANDOM_STATE = 42
# ---------------

def clean_and_prepare(df):
    # Normalizza nomi colonne
    df.columns = df.columns.str.strip()

    # Rimuovi righe che sembrano header duplicati
    df = df[df['season'] != 'stagione']
    
    # Standardizza team name (semplice)
    df = df.copy()  # evita warning view/copy
    df['squad'] = df['squad'].astype(str).str.strip()
    
    # Standardize team names
    team_name_map = {
        'Inter Milan': 'Inter',
        'AC Milan': 'Milan',
        'Internazionale': 'Inter',
        'Hellas': 'Hellas Verona',
        'Verona': 'Hellas Verona',
    }
    df['squad'] = df['squad'].replace(team_name_map)
    
    # Remove any rows where squad is empty or null
    df = df[df['squad'].notna() & (df['squad'] != '')]

    # Rimozione colonne di leakage/identificatori e statistiche avanzate che potrebbero causare leakage
    drop_cols = [
        # Identificatori e meta-dati
       
    ]
    
    print("\nRimozione colonne per evitare data leakage:")
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
            print(f"- Rimossa colonna: {c}")
            
    # Mostra le colonne rimaste per verifica
    print("\nColonne utilizzate per il modello:")
    print(df.columns.tolist())
            
    # Rimuovi colonne completamente vuote
    empty_cols = [col for col in df.columns 
                 if df[col].isna().all() or (df[col] == '').all()]
    if empty_cols:
        print(f"[INFO] Rimozione colonne vuote: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # Crea target binario se non esiste
    if 'winner' not in df.columns:
        # assume Rk era presente: fallback, ma tu hai già winner creato
        raise ValueError("Colonna 'winner' non trovata. Creane una dove winner=1 per Rk==1.")

    # Assicura tipi numerici per colonne potenzialmente numeriche
    # prova a convertire tutte le colonne tranne 'squad' e 'season' e 'winner'
    non_num = ['squad', 'season']
    for col in df.columns:
        if col in non_num or col == 'winner':
            continue
        # rimuovi percentuali e simboli, sostituisci virgola con punto
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('%','', regex=False).str.replace(',','.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Controllo duplicati per (squad,season)
    dup_counts = df.groupby(['squad','season']).size().value_counts()
    if dup_counts.index.max() > 1:
        # Aggrega: per stats continui usiamo media, per count potresti usare sum
        # Qui facciamo una media generica (modifica se vuoi somme per alcune colonne)
        df = df.groupby(['squad','season'], as_index=False).mean()
        # winner è lost via mean -> ricostruisci winner dal Rank se disponibile,
        # qui assumiamo che winner era 0/1 e la media ricade a 1 o 0
        if 'winner' in df.columns:
            df['winner'] = (df['winner'] >= 0.5).astype(int)

    return df

def split_train_predict(df):
    # Assumi season come string "2010-11" o numero; estrai anno finale per split
    # Convert season to year_end integer in a robust way
    def season_to_year(s):
        try:
            s = str(s)
            if '-' in s:
                parts = s.split('-')
                # prendi la seconda parte, se '10-11' -> '11' -> 2011 or 2011 if 20xx etc.
                y = parts[-1]
                if len(y) == 2:
                    # map '10' -> 2010? We'll assume seasons 2010-2025 -> if y >= 0 and <= 25
                    yint = int(y)
                    if yint <= 25:
                        return 2000 + yint
                    else:
                        return 1900 + yint
                else:
                    return int(y)
            else:
                return int(s)
        except:
            return np.nan

    df['season_year_end'] = df['season'].apply(season_to_year)
    
    print("\nSplit info:")
    print(f"Total rows: {len(df)}")
    print(f"Available years: {sorted(df['season_year_end'].unique())}")
    print(f"Training up to: {TRAIN_UP_TO_YEAR}")
    print(f"Predicting for: {PREDICT_YEAR}")

    train_df = df[df['season_year_end'] <= TRAIN_UP_TO_YEAR].copy()
    predict_df = df[df['season_year_end'] == PREDICT_YEAR].copy()
    
    print(f"Training rows: {len(train_df)}")
    print(f"Prediction rows: {len(predict_df)}")

    if train_df.shape[0] == 0:
        raise ValueError("Nessun training data trovato per il range definito.")

    return train_df, predict_df
def build_features_targets(train_df):
    # Aggiungi feature ingegnerizzate
    df = train_df.copy()
    
    # Ensure non-zero denominators
    eps = 1e-8  # small constant to avoid division by zero
    safe_90s = df['90s'].astype(float).replace(0, eps)
    safe_shots = df['Sh'].astype(float).replace(0, eps)
    safe_sot = df['SoT'].astype(float).replace(0, eps)
    safe_goals = df['Gls'].astype(float).replace(0, eps)
    
    # Features basate sui gol e assist
    df['goals_per_game'] = df['Gls'].astype(float) / safe_90s
    df['goals_assists_per_game'] = df['G+A'].astype(float) / safe_90s
    df['non_penalty_goals_per_game'] = df['G-PK'].astype(float) / safe_90s
    
    # Shooting efficiency
    df['shot_accuracy'] = df['SoT'].astype(float) / safe_shots
    df['shots_per_game'] = df['Sh'].astype(float) / safe_90s
    df['shots_on_target_per_game'] = df['SoT'].astype(float) / safe_90s
    df['goals_per_shot'] = df['Gls'].astype(float) / safe_shots
    df['goals_per_shot_on_target'] = df['Gls'].astype(float) / safe_sot
    
    # Penalty performance
    df['penalty_conversion'] = df['PK'].astype(float) / df['PKatt'].astype(float).replace(0, eps)
    
    # Game time utilization
    df['minutes_per_game'] = df['Min'].astype(float) / df['MP'].astype(float).replace(0, eps)
    df['starts_ratio'] = df['Starts'].astype(float) / df['MP'].astype(float).replace(0, eps)
    
    # Assist metrics
    df['assists_per_game'] = df['Ast'].astype(float) / safe_90s
    df['assist_to_goal_ratio'] = df['Ast'].astype(float) / safe_goals
    
    # Possession and Control
    df['possession_score'] = df['Poss'].astype(float)
    
    # Calcola medie di squadra per season
    season_avg = df.groupby('season').agg({
        'goals_per_game': 'mean',
        'shots_per_game': 'mean',
        'possession_score': 'mean'
    }).reset_index()
    
    # Unisci con il dataframe originale
    df = df.merge(season_avg, on='season', suffixes=('', '_season_avg'))
    
    # Crea feature relative
    df['goals_vs_avg'] = df['goals_per_game'] / df['goals_per_game_season_avg']
    df['shots_vs_avg'] = df['shots_per_game'] / df['shots_per_game_season_avg']
    df['possession_vs_avg'] = df['possession_score'] / df['possession_score_season_avg']

    # Drop base columns no longer needed
    drop_cols = ['season', 'season_year_end']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df.drop(columns=['squad', 'winner'])
    y = df['winner'].astype(int)

    # Rimuovi colonne con tutti NA
    na_cols = X.columns[X.isna().all()]
    if len(na_cols) > 0:
        print(f"[INFO] Rimozione colonne con tutti NA: {na_cols}")
        X = X.drop(columns=na_cols)

    # Imputers + scaler
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns, index=X.index)

    return X_scaled, y, imputer, scaler

def train_xgboost(X, y):
    # Bilanciamento più moderato per evitare overconfidence
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = max(3.0, n_neg / max(1.0, n_pos))  # ridotto per più balance
    
    print(f"\nTraining info:")
    print(f"Positive examples (winners): {n_pos}")
    print(f"Negative examples: {n_neg}")
    print(f"Scale pos weight: {scale_pos_weight}")

    # Split per validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=2000,        # ridotto per evitare overfitting
        learning_rate=0.03,      # aumentato per probabilità più alte
        max_depth=5,              # ridotto per migliore generalizzazione
        min_child_weight=4,       # aumentato per stabilità
        subsample=0.7,            # ridotto per più variabilità
        colsample_bytree=0.8,     # ridotto per più variabilità
        colsample_bylevel=0.8,    # ridotto per più variabilità
        gamma=0.1,                # ridotto per split più fini
        scale_pos_weight=scale_pos_weight * 1.2,  # aumentato per dare più peso ai vincitori
        random_state=RANDOM_STATE
    )

    # Training con early stopping usando callbacks
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )
    
    return model

    model.fit(X, y, verbose=True)
    return model

def evaluate_walk_forward(df):
    # Rimuovi colonne con tutti NA una volta sola all'inizio
    feature_cols = [c for c in df.columns if c not in ['squad','season','season_year_end','winner']]
    na_cols = [c for c in feature_cols if df[c].isna().all()]
    if na_cols:
        print(f"[INFO] Rimozione colonne NA dalla valutazione: {na_cols}")
        df = df.drop(columns=na_cols)
    
    # Walk-forward: per ogni year t in [start+? .. TRAIN_UP_TO_YEAR-1], train <= t, test t+1
    years = sorted(df['season_year_end'].unique())
    years = [y for y in years if y <= TRAIN_UP_TO_YEAR]
    results = []
    for i in range(3, len(years)-0):  # comincia quando hai almeno 3 stagioni per train
        train_years = years[:i+1]  # upto year[i]
        test_year = years[i+1] if i+1 < len(years) else None
        if test_year is None or test_year > TRAIN_UP_TO_YEAR:
            break
        train_df = df[df['season_year_end'].isin(train_years)]
        test_df = df[df['season_year_end']==test_year]
        if test_df.empty: continue

        X_train = train_df.drop(columns=['squad','season','season_year_end','winner'])
        y_train = train_df['winner'].astype(int)
        X_test = test_df.drop(columns=['squad','season','season_year_end','winner'])
        y_test = test_df['winner'].astype(int)

        # imputazione/scaling (fit on train only)
        imp = SimpleImputer(strategy='median').fit(X_train)
        sc = StandardScaler().fit(imp.transform(X_train))
        Xtr = pd.DataFrame(sc.transform(imp.transform(X_train)), columns=X_train.columns)
        Xte = pd.DataFrame(sc.transform(imp.transform(X_test)), columns=X_test.columns)

        # train XGB con stessi parametri del modello finale
        pos = y_train.sum(); neg = len(y_train)-pos
        scale_pos_weight = max(3.0, neg / max(1.0, pos))
        clf = xgb.XGBClassifier(
            n_estimators=1500,        # ridotto per evitare overfitting
            learning_rate=0.02,       # aumentato per convergenza più rapida
            max_depth=4,              # ridotto per maggiore generalizzazione
            min_child_weight=6,       # aumentato per ridurre overfitting
            subsample=0.85,           # aumentato per stabilità
            colsample_bytree=0.85,    # aumentato per stabilità
            colsample_bylevel=0.85,   # aumentato per stabilità
            gamma=0.2,                # aumentato per ridurre overfitting
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE
        )
        clf.fit(Xtr, y_train)

        y_proba = clf.predict_proba(Xte)[:,1]
        # metriche: AUC e Top-1 accuracy
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = np.nan
        # top-1 per stagione test (solo 1 stagione quindi)
        # seleziona team con massima proba nella season e verifica se è il vincitore
        top_idx = np.argmax(y_proba)
        top_hit = int(y_test.values[top_idx] == 1)
        brier = brier_score_loss(y_test, y_proba)
        results.append({'train_up_to': train_years[-1], 'test_year': test_year,
                        'auc': auc, 'top1': top_hit, 'brier': brier})
    return pd.DataFrame(results)

def create_features(df, season_stats=None):
    """Helper function to create consistent features for both training and prediction"""
    # Convert season to numeric year for temporal features
    def extract_year(season):
        try:
            if '-' in str(season):
                return int('20' + str(season).split('-')[1])
            return int(season)
        except:
            return None
    
    df['year'] = df['season'].apply(extract_year)
    
    # Ensure non-zero denominators
    eps = 1e-8
    safe_90s = df['90s'].astype(float).replace(0, eps)
    safe_shots = df['Sh'].astype(float).replace(0, eps)
    safe_sot = df['SoT'].astype(float).replace(0, eps)
    safe_goals = df['Gls'].astype(float).replace(0, eps)
    
    # Features basate sui gol e assist
    df['goals_per_game'] = df['Gls'].astype(float) / safe_90s
    df['goals_assists_per_game'] = df['G+A'].astype(float) / safe_90s
    df['non_penalty_goals_per_game'] = df['G-PK'].astype(float) / safe_90s
    
    # Shooting efficiency
    df['shot_accuracy'] = df['SoT'].astype(float) / safe_shots
    df['shots_per_game'] = df['Sh'].astype(float) / safe_90s
    df['shots_on_target_per_game'] = df['SoT'].astype(float) / safe_90s
    df['goals_per_shot'] = df['Gls'].astype(float) / safe_shots
    df['goals_per_shot_on_target'] = df['Gls'].astype(float) / safe_sot
    
    # Penalty performance
    df['penalty_conversion'] = df['PK'].astype(float) / df['PKatt'].astype(float).replace(0, eps)
    
    # Game time utilization
    df['minutes_per_game'] = df['Min'].astype(float) / df['MP'].astype(float).replace(0, eps)
    df['starts_ratio'] = df['Starts'].astype(float) / df['MP'].astype(float).replace(0, eps)
    
    # Assist metrics
    df['assists_per_game'] = df['Ast'].astype(float) / safe_90s
    df['assist_to_goal_ratio'] = df['Ast'].astype(float) / safe_goals
    
    # Possession and Control
    df['possession_score'] = df['Poss'].astype(float)
    
    if season_stats is None:
        # Calculate robust statistics using median and quantiles
        season_stats = df.groupby('season').agg({
            'goals_per_game': ['median', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25)],
            'shots_per_game': ['median', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25)],
            'possession_score': ['median', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25)]
        }).reset_index()
        
        # Flatten column names
        season_stats.columns = ['season', 
                              'goals_per_game_median', 'goals_per_game_75', 'goals_per_game_25',
                              'shots_per_game_median', 'shots_per_game_75', 'shots_per_game_25',
                              'possession_score_median', 'possession_score_75', 'possession_score_25']
    
    # Calculate simple season averages for relative features
    season_means = df.groupby('season').agg({
        'goals_per_game': 'mean',
        'shots_per_game': 'mean',
        'possession_score': 'mean'
    }).reset_index()
    season_means.columns = ['season', 
                          'goals_per_game_season_avg',
                          'shots_per_game_season_avg', 
                          'possession_score_season_avg']
    
    # Merge season averages
    df = df.merge(season_means, on='season')
    
    # Create relative features
    df['goals_vs_avg'] = df['goals_per_game'] / df['goals_per_game_season_avg']
    df['shots_vs_avg'] = df['shots_per_game'] / df['shots_per_game_season_avg']
    df['possession_vs_avg'] = df['possession_score'] / df['possession_score_season_avg']
    
    # Add position relative to median and quartiles if provided
    if season_stats is not None and len(season_stats.columns) > 4:  # if we have quartile stats
        df = df.merge(season_stats, on='season')
        # Add features relative to distribution
        df['goals_vs_median'] = df['goals_per_game'] / df['goals_per_game_median']
        df['goals_percentile'] = ((df['goals_per_game'] - df['goals_per_game_25']) / 
                                (df['goals_per_game_75'] - df['goals_per_game_25']))
        df['shots_percentile'] = ((df['shots_per_game'] - df['shots_per_game_25']) / 
                               (df['shots_per_game_75'] - df['shots_per_game_25']))
        df['possession_percentile'] = ((df['possession_score'] - df['possession_score_25']) / 
                                    (df['possession_score_75'] - df['possession_score_25']))
    
    # Add historical performance indicators with exponential weighting
    recent_seasons = df.sort_values('year', ascending=True)
    teams = df['squad'].unique()
    
    # Initialize historical features
    df['historical_goals_ratio'] = 1.0
    df['historical_win_rate'] = 0.0
    df['years_since_winner'] = 10.0  # default to 10 years if no history
    df['recent_performance'] = 0.0    # New feature for recent performance
    df['historical_top3_rate'] = 0.0  # New feature for top 3 finishes
    
    for team in teams:
        team_data = recent_seasons[recent_seasons['squad'] == team]
        if len(team_data) > 1:
            # Calculate recent performance with stronger exponential weighting
            team_data = team_data.sort_values('year', ascending=True)
            weights = np.exp(np.linspace(-2, 0, len(team_data)))  # Reduced decay rate
            weights = weights / weights.sum()  # Normalize weights
            
            # Calculate weighted historical goals ratio
            goals_ratio = (team_data['goals_per_game'] / team_data['goals_per_game_season_avg']).values
            hist_goals_ratio = np.sum(goals_ratio * weights)
            df.loc[df['squad'] == team, 'historical_goals_ratio'] = hist_goals_ratio
            
            # Calculate weighted win rate
            win_rate = np.sum(team_data['winner'].values * weights)
            df.loc[df['squad'] == team, 'historical_win_rate'] = win_rate
            
            # Calculate recent performance (last 3 seasons)
            last_seasons = team_data.tail(3)
            if len(last_seasons) > 0:
                recent_weights = np.exp(np.linspace(-2, 0, len(last_seasons)))
                recent_weights = recent_weights / recent_weights.sum()
                recent_goals_ratio = (last_seasons['goals_per_game'] / last_seasons['goals_per_game_season_avg']).values
                recent_perf = np.sum(recent_goals_ratio * recent_weights)
                df.loc[df['squad'] == team, 'recent_performance'] = recent_perf
            
            # Calculate historical top 3 rate (teams that finished in top 3)
            top3_rate = ((team_data['goals_vs_avg'] >= team_data.groupby('season')['goals_vs_avg'].transform('quantile', 0.85)).mean())
            df.loc[df['squad'] == team, 'historical_top3_rate'] = top3_rate
            
            # Years since last title with slower exponential decay
            if team_data['winner'].sum() > 0:
                last_win_year = team_data[team_data['winner'] == 1]['year'].max()
                years_since = df['year'] - last_win_year
                df.loc[df['squad'] == team, 'years_since_winner'] = np.exp(-years_since/8)  # Slower decay with 8-year half-life
            
            # Add recent title contender status (top 3 finishes in last 3 years)
            recent_seasons = team_data.tail(3)
            if len(recent_seasons) > 0:
                recent_top3 = (recent_seasons['goals_vs_avg'] >= recent_seasons.groupby('season')['goals_vs_avg'].transform('quantile', 0.85)).mean()
                df.loc[df['squad'] == team, 'recent_contender'] = recent_top3
    
    return df

def build_features_targets(train_df):
    df = create_features(train_df.copy())
    
    # Drop unnecessary columns
    drop_cols = ['season', 'season_year_end']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    X = df.drop(columns=['squad', 'winner'])
    y = df['winner'].astype(int)
    
    # Handle NA columns
    na_cols = X.columns[X.isna().all()]
    if len(na_cols) > 0:
        print(f"[INFO] Rimozione colonne con tutti NA: {na_cols}")
        X = X.drop(columns=na_cols)
    
    # Imputers + scaler
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns, index=X.index)
    
    return X_scaled, y, imputer, scaler

def predict_for_year(model, imputer, scaler, predict_df, train_df):
    print(f"\nPredict year info:")
    print(f"Total rows: {len(predict_df)}")
    print(f"Years in data: {sorted(predict_df['season_year_end'].unique())}")
    print(f"Prediction year: {PREDICT_YEAR}")
    
    # Use training data as historical data for normalization
    historical_df = train_df.copy()  # Use all training data for historical averages
    current_df = predict_df[predict_df['season_year_end'] == PREDICT_YEAR].copy()
    
    print(f"Historical rows: {len(historical_df)}")
    print(f"Current year rows: {len(current_df)}")
    
    if len(current_df) == 0:
        print(f"\nERROR: No data found for prediction year {PREDICT_YEAR}")
        print("Available years:", sorted(predict_df['season_year_end'].unique()))
        return None
        
    # Use training data averages if no historical data available
    if len(historical_df) == 0:
        print("\nWARNING: No historical data available. Using training data averages for normalization.")
        # Load training data
        try:
            train_df = pd.read_csv(DATA_CSV)
            train_df = clean_and_prepare(train_df)
            # Convert season to year_end format
            def season_to_year(s):
                try:
                    s = str(s)
                    if '-' in s:
                        year = int('20' + s.split('-')[1])
                        return year
                    return int(s)
                except:
                    return None
            
            train_df['season_year_end'] = train_df['season'].apply(season_to_year)
            train_df = train_df[train_df['season_year_end'] <= TRAIN_UP_TO_YEAR]
            if len(train_df) > 0:
                historical_df = train_df.copy()
                print(f"Loaded {len(historical_df)} historical records from training data for normalization")
            else:
                raise ValueError("No historical data found in training set")
        except Exception as e:
            print(f"Could not load training data: {e}")
            print("Using current year data for normalization")
            historical_df = current_df.copy()
    
    # Create basic features first (without season averages)
    eps = 1e-8
    for df in [historical_df, current_df]:
        if len(df) > 0:  # Only process if we have data
            safe_90s = df['90s'].astype(float).replace(0, eps)
            safe_shots = df['Sh'].astype(float).replace(0, eps)
            
            # Basic per-game stats needed for season averages
            df['goals_per_game'] = df['Gls'].astype(float) / safe_90s
            df['shots_per_game'] = df['Sh'].astype(float) / safe_90s
            df['possession_score'] = df['Poss'].astype(float)
    
    # Now calculate season stats from historical data
    if len(historical_df) > 0:
        season_stats = historical_df.groupby('season').agg({
            'goals_per_game': 'mean',
            'shots_per_game': 'mean',
            'possession_score': 'mean'
        }).reset_index()
    else:
        # If no historical data, use current year's averages
        season_stats = current_df.groupby('season').agg({
            'goals_per_game': 'mean',
            'shots_per_game': 'mean',
            'possession_score': 'mean'
        }).reset_index()
    
    # Create all features including relative ones
    df = create_features(current_df.copy(), season_stats)
    Xp = df.drop(columns=['squad', 'season', 'season_year_end', 'winner'])
    
    # Ensure columns match training data
    missing_cols = set(imputer.feature_names_in_) - set(Xp.columns)
    for col in missing_cols:
        Xp[col] = 0  # Add missing columns with neutral values
    
    Xp = Xp[imputer.feature_names_in_]  # Ensure same column order as training
    
    # Apply same transformations as training
    Xp_imp = pd.DataFrame(imputer.transform(Xp), columns=Xp.columns, index=Xp.index)
    Xp_scaled = pd.DataFrame(scaler.transform(Xp_imp), columns=Xp.columns, index=Xp.index)
    
    # Get predictions
    probs = model.predict_proba(Xp_scaled)[:,1]
    
    # Use softmax with temperature for probability calibration
    def softmax_with_temp(x, temperature=2.5):
        # Higher temperature = softer distribution, lower = sharper
        exp_x = np.exp((x * 10) / temperature)  # Scale up probabilities before temperature scaling
        return exp_x / exp_x.sum()
    
    # Apply calibration with temperature
    calibrated_probs = softmax_with_temp(probs)
    
    # Create output dataframe
    out = predict_df[['squad', 'season']].copy()
    out['prob_win'] = calibrated_probs
    
    # Remove any duplicates and sort
    out = out.drop_duplicates(subset=['squad'], keep='first')
    out = out.sort_values('prob_win', ascending=False).reset_index(drop=True)
    
    # Remove any duplicates (keep first occurrence)
    out = out.drop_duplicates(subset=['squad'], keep='first').copy()
    out = out.sort_values('prob_win', ascending=False).reset_index(drop=True)
    
    # Add rank info
    out['rank'] = range(1, len(out) + 1)
    
    return out

def create_analysis_plots(train_df, X_train, y_train, model, wfv, artifacts_dir):
    # 1. Performance over time plot
    plt.figure(figsize=(12, 6))
    plt.plot(wfv['test_year'], wfv['auc'], 'b-o', label='AUC')
    plt.plot(wfv['test_year'], wfv['top1'], 'r-o', label='Top-1 Accuracy')
    plt.title('Model Performance Over Time')
    plt.xlabel('Test Year')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(artifacts_dir, 'performance_over_time.png'))
    plt.close()

    # Create multiple pairplots for different feature groups to avoid overcrowding
    # Create DataFrame with all features
    pair_df = pd.DataFrame(X_train, columns=X_train.columns)
    pair_df['winner'] = y_train
    
    # Group 1: Goal-related features (top importance)
    goal_features = ['goals_percentile', 'goals_vs_avg', 'goals_per_shot_on_target', 'G/SoT', 'winner']
    sns.pairplot(pair_df[goal_features], hue='winner', diag_kind='kde')
    plt.suptitle('Goal-Related Features Relationships', y=1.02)
    plt.savefig(os.path.join(artifacts_dir, 'goal_features_relationships.png'))
    plt.close()
    
    # Group 2: Shot-related features
    shot_features = ['SoT', 'shot_accuracy', 'shots_vs_avg', 'SoT/90', 'winner']
    sns.pairplot(pair_df[shot_features], hue='winner', diag_kind='kde')
    plt.suptitle('Shot-Related Features Relationships', y=1.02)
    plt.savefig(os.path.join(artifacts_dir, 'shot_features_relationships.png'))
    plt.close()
    
    # Group 3: Possession and general play features
    play_features = ['possession_percentile', 'Poss', 'assist_to_goal_ratio', 'penalty_conversion', 'winner']
    sns.pairplot(pair_df[play_features], hue='winner', diag_kind='kde')
    plt.suptitle('Possession and Play Features Relationships', y=1.02)
    plt.savefig(os.path.join(artifacts_dir, 'play_features_relationships.png'))
    plt.close()

    # 2. Feature correlations heatmap
    plt.figure(figsize=(15, 12))
    corr = X_train.corr()
    sns.heatmap(corr, cmap='RdBu', center=0, annot=False)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'feature_correlations.png'))
    plt.close()

    # 3. Distribution of key features for winners vs non-winners
    key_features = ['goals_per_game', 'possession_score', 'shots_per_game']
    fig, axes = plt.subplots(len(key_features), 1, figsize=(12, 4*len(key_features)))
    for i, feature in enumerate(key_features):
        if feature in train_df.columns:
            sns.boxplot(x='winner', y=feature, data=train_df, ax=axes[i])
            axes[i].set_title(f'{feature} Distribution by Winner Status')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'feature_distributions.png'))
    plt.close()

    # 4. Model learning curves
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc'
    )
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'b-', label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'r-', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('ROC AUC Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(artifacts_dir, 'learning_curves.png'))
    plt.close()

    # 5. Prediction confidence distribution
    predictions = model.predict_proba(X_train)[:, 1]
    plt.figure(figsize=(10, 6))
    plt.hist(predictions[y_train==0], bins=50, alpha=0.5, label='Non-Winners', density=True)
    plt.hist(predictions[y_train==1], bins=50, alpha=0.5, label='Winners', density=True)
    plt.xlabel('Predicted Probability of Winning')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.savefig(os.path.join(artifacts_dir, 'prediction_distribution.png'))
    plt.close()

def main():
    # Import required libraries at the top
    import seaborn as sns
    
    df = pd.read_csv(DATA_CSV)
    df = clean_and_prepare(df)
    train_df, predict_df = split_train_predict(df)

    # walk-forward eval (stima storica di performance)
    print("Eseguo walk-forward evaluation...")
    wfv = evaluate_walk_forward(train_df)
    print(wfv)

    # build features/targets su tutto train set
    X_train, y_train, imputer, scaler = build_features_targets(train_df)

    # Crea cartella artifacts
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # train finale
    print("Addestramento finale su 2010-2020...")
    model = train_xgboost(X_train, y_train)

    # Analisi SHAP per interpretabilità
    print("Calcolo SHAP values per interpretabilità...")
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Plot SHAP summary
    print("Genero plot SHAP...")
    plt.figure(figsize=(10, 8))  # dimensione figura più grande
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()  # migliore layout
    plt.savefig(os.path.join(artifacts_dir, 'shap_summary.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Feature importance basata su SHAP
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv(os.path.join(artifacts_dir, 'feature_importance.csv'), index=False)
    print("\nTop 10 features più importanti:")
    print(feature_importance.head(10).to_string(index=False))
    if predict_df.shape[0] == 0:
        print("Nessuna riga per predictive year found.")
    else:
        out = predict_for_year(model, imputer, scaler, predict_df, train_df)
        
        # Plot predictions visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(x='prob_win', y='squad', data=out.head(10), hue='squad', legend=False)
        plt.title(f'Top 10 Teams - Win Probability for {PREDICT_YEAR}')
        plt.xlabel('Probability of Winning')
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, 'predictions_visualization.png'))
        plt.close()
        
        print(f"\nPredizioni Serie A {PREDICT_YEAR}")
        print("\nProbabilità di vittoria del campionato:")
        # Format probabilities as percentages
        out['probability'] = (out['prob_win'] * 100).round(2).astype(str) + '%'
        print(out[['rank', 'squad', 'probability']].to_string(index=False))
    # Generate analysis plots
    print("\nGenerating analysis plots...")
    create_analysis_plots(train_df, X_train, y_train, model, wfv, artifacts_dir)

    # salvataggi modello
    joblib.dump(model, os.path.join(artifacts_dir, "xgb_winner_model.joblib"))
    joblib.dump(imputer, os.path.join(artifacts_dir, "imputer.joblib"))
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))
    print("\nModello e analisi salvati in artifacts/")

    # predici 2021
   

if __name__ == "__main__":
    main()
