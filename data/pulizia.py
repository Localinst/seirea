# prepare_dataset_from_raw.py
import pandas as pd
import numpy as np
import re

RAW_CSV = 'C:/Users/ReadyToUse/Desktop/Data/serie a/data/combined_all.csv'   # file con righe come l'esempio
OUT_CSV = "dataset_cleaned_for_ml.csv"

# Lista delle colonne nell'ordine che hai fornito (adatta se l'ordine differisce)
cols = [
    "season","squad","winner","# Pl","Age","Poss","MP","Starts","Min","90s",
    "Gls","Ast","G+A","G-PK","PK","PKatt","CrdY","CrdR","G+A-PK",
    "Rk","W","D","L","GF","GA","GD","Pts","Pts/MP","Attendance",
    "Top Team Scorer","Goalkeeper","Notes","Sh","SoT","SoT%","Sh/90","SoT/90",
    "G/Sh","G/SoT","Dist","xG","xGA","xGD","xGD/90"
]

def read_raw(filepath):
    # prova a leggere con pandas; se header mancante, leggi senza header
    try:
        df = pd.read_csv(filepath, header=None, dtype=str, encoding='utf-8', keep_default_na=False)
    except Exception:
        df = pd.read_csv(filepath, header=None, dtype=str, encoding='latin1', keep_default_na=False)
    # se il numero di colonne non corrisponde a cols, provvedi ad adattare
    if df.shape[1] != len(cols):
        # cerca pattern comune: se ci sono colonne in più o in meno, avvisa
        print(f"[WARN] Col count mismatch: file has {df.shape[1]} cols, expected {len(cols)}.")
        # procediamo comunque: tronca o aggiungi NaN
        if df.shape[1] > len(cols):
            df = df.iloc[:, :len(cols)]
        else:
            for i in range(len(cols) - df.shape[1]):
                df[i+df.shape[1]] = ""
    df.columns = cols
    return df

# pulizia generale di numerici: rimuove % e sostituisce virgola decimale
def clean_numeric_col(s: pd.Series) -> pd.Series:
    # Prima converti tutto in stringhe, gestendo NaN/None
    s = s.fillna('').astype(str)
    
    # Rimuovi spazi e caratteri speciali solo per stringhe non vuote
    mask = s != ''
    if mask.any():
        s.loc[mask] = s.loc[mask].str.strip()
        # rimuovi % e spazi
        s.loc[mask] = s.loc[mask].str.replace('%', '', regex=False)
        s.loc[mask] = s.loc[mask].str.replace(r'\s+', '', regex=True)
        # sostituisci comma decimale
        s.loc[mask] = s.loc[mask].str.replace(',', '.', regex=False)
        # rimuovi caratteri non numerici (eccetto - e .)
        s.loc[mask] = s.loc[mask].str.replace(r'[^\d\.\-]', '', regex=True)
    
    # Converti stringhe vuote in NaN
    s = s.replace('', np.nan)
    
    # Converti in numerico
    return pd.to_numeric(s, errors='coerce')
def prepare(df):
    # rimuovi BOM/spazi in squad
    df['squad'] = df['squad'].astype(str).str.strip()

    # converti colonne numeriche conosciute
    numeric_cols = [c for c in cols if c not in ['season','squad','Top Team Scorer','Goalkeeper','Notes']]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = clean_numeric_col(df[c])

    # se winner non esiste o è NaN, prova a ricostruirla da Rk
    if 'winner' in df.columns:
        df['winner'] = df['winner'].fillna(0).astype(int)
    elif 'Rk' in df.columns:
        df['winner'] = (df['Rk'] == 1).astype(int)
    else:
        raise ValueError("Né 'winner' né 'Rk' trovati; serve uno dei due.")

    # Normalizza season: trim e uniforma tipo '2010-2011'
    df['season'] = df['season'].astype(str).str.strip()

    # Controlla duplicati per (squad, season)
    dup = df.groupby(['squad','season']).size()
    dup_non1 = dup[dup > 1]
    if not dup_non1.empty:
        print("[INFO] Trovati duplicate squad-season. Effettuo aggregazione media su colonne numeriche.")
        # Aggrega: media per numeriche, per colonne testuali prendi il primo non-null
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = [c for c in df.columns if c not in num_cols]
        agg_num = df.groupby(['squad','season'])[num_cols].mean().reset_index()
        # per i testuali prendo il primo
        agg_txt = df.groupby(['squad','season'])[txt_cols].first().reset_index()
        df = pd.merge(agg_txt, agg_num, on=['squad','season'], how='inner')

    # Rimuovi colonne di leakage (come concordato)
    to_drop = ['Rk','Pts','Pts/MP','W','D','L','GF','GA','GD','Top Team Scorer','Goalkeeper','Notes']
    for c in to_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Opzionale: riempi NaN numerici con median (imputation semplice)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        med = df[c].median(skipna=True)
        df[c] = df[c].fillna(med)

    # Ordina colonne finali utili
    final_cols = ['season','squad','winner'] + [c for c in df.columns if c not in ['season','squad','winner']]
    df = df[final_cols]
    return df

if __name__ == "__main__":
    df_raw = read_raw(RAW_CSV)
    print(f"[INFO] Read raw rows: {len(df_raw)}")
    df_clean = prepare(df_raw)
    print("[INFO] Cleaned dataset shape:", df_clean.shape)
    df_clean.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved cleaned dataset to {OUT_CSV}")
