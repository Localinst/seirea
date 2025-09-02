import os
import time
import sys
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def cols_to_strings(df):
    # flatten MultiIndex columns to readable strings
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            cols.append(' '.join([str(x) for x in c if x and str(x) != '']))
        else:
            cols.append(str(c))
    return cols

def has_substrs(df, substrs):
    cols = cols_to_strings(df)
    return all(any(sub in col for col in cols) for sub in substrs)

def save_df(df, path):
    try:
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print('WROTE', path)
    except Exception as e:
        print('ERROR writing', path, e, file=sys.stderr)

for x in range(15):
    anno1 = 2024+ x
    anno2 = 2025 + x
    season = f'{anno1}-{anno2}'
    url = f'https://fbref.com/en/comps/11/{season}/{season}-Serie-A-Stats'
    print('\nProcessing', season, '->', url)
    try:
        tables = pd.read_html(url)
    except Exception as e:
        print('SKIP', season, 'read_html failed:', e, file=sys.stderr)
        # wait a bit and continue
        time.sleep(1)
        continue

    
    results = None
   

    # attempt to find tables by expected column substrings
    for t in tables:
        try:
            
            if results is None and (has_substrs(t, ['Age'])):
                results = t
                continue
        except Exception:
            # defensive: if a table has weird columns, skip
            continue

    # Fallbacks: if not found by heuristic, try the ids used on fbref pages
    if results is None:
        try:
            std_tables = pd.read_html(url, attrs={"id":"stats_squads_standard_for"})
            if std_tables:
                results = std_tables[0]
        except Exception:
            pass
   
    # Save found tables
    
    if results is not None:
        save_df(results, os.path.join(DATA_DIR, f'{season}_resultss.csv'))
    else:
        print('results table not found for', season, file=sys.stderr)

   
    

    # polite delay
    time.sleep(1)
       
    