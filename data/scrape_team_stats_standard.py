"""Scrape team pages and extract the table with id 'stats_standard_11'.

Behavior:
- Look for files in the repo data/ directory named like '<season>_link.csv' (created by main.py).
- For each file from season >= 2010-2011, read `squad_href` column and fetch each team URL.
- Find the HTML <table id="stats_standard_11"> (or inside HTML comments) and save it as CSV under
  data/players/{season}/{season}_{team_safe}_stats_standard_11.csv

Usage:
    python data/scrape_team_stats_standard.py

Environment:
  - RATE_SLEEP (seconds) default 1
  - START_YEAR (numeric) default 2010
  - USER_AGENT override
"""
import os
import time
import glob
import sys
from urllib.parse import urljoin

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup, Comment
except Exception:
    BeautifulSoup = None
    Comment = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
PLAYERS_DIR = os.path.join(DATA_DIR, 'players')
os.makedirs(PLAYERS_DIR, exist_ok=True)

RATE_SLEEP = float(os.environ.get('RATE_SLEEP', '20'))
START_YEAR = int(os.environ.get('START_YEAR', '2024'))
USER_AGENT = os.environ.get('USER_AGENT', 'python-requests/0.0')


def safe_name(s: str) -> str:
    return ''.join([c if c.isalnum() or c in (' ', '-', '_') else '_' for c in s]).strip().replace(' ', '_')


def find_table_in_comments(soup, table_id):
    # Some FBref tables are inside HTML comments; search comments for a matching table
    if Comment is None:
        return None
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        text = str(comment)
        if f'id="{table_id}"' in text or f"id='{table_id}'" in text:
            try:
                tbl = BeautifulSoup(text, 'html.parser').find('table', id=table_id)
                if tbl is not None:
                    return tbl
            except Exception:
                continue
    return None


def extract_and_save(team_url, season, squad_name):
    headers = {'User-Agent': USER_AGENT}
    for attempt in range(3):
        try:
            r = requests.get(team_url, headers=headers, timeout=15)
            r.raise_for_status()
            html = r.text
            if BeautifulSoup:
                soup = BeautifulSoup(html, 'html.parser')
                tbl = soup.find('table', id='stats_standard_11')
                if tbl is None:
                    tbl = find_table_in_comments(soup, 'stats_standard_11')
                if tbl is not None:
                    try:
                        df = pd.read_html(str(tbl))[0]
                    except Exception:
                        df = None
                else:
                    # last-resort: try pandas on full HTML for a table with Player column
                    try:
                        all_tables = pd.read_html(html)
                        df = None
                        for t in all_tables:
                            cols = [str(c).lower() for c in t.columns]
                            if any('player' in c for c in cols) or 'pos' in cols:
                                df = t
                                break
                    except Exception:
                        df = None
            else:
                # BeautifulSoup not available - try pandas directly
                try:
                    all_tables = pd.read_html(html)
                    df = None
                    for t in all_tables:
                        cols = [str(c).lower() for c in t.columns]
                        if any('player' in c for c in cols) or 'pos' in cols:
                            df = t
                            break
                except Exception:
                    df = None

            if df is not None and not df.empty:
                season_safe = season.replace('/', '-').replace(' ', '_')
                team_safe = safe_name(squad_name)
                out_dir = os.path.join(PLAYERS_DIR, season)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{season_safe}_{team_safe}_stats_standard_11.csv")
                df.to_csv(out_path, index=False, encoding='utf-8-sig')
                print('WROTE', out_path)
                return True
            else:
                print(f'WARN: table not found for {squad_name} at {team_url} (attempt {attempt+1})', file=sys.stderr)
        except Exception as e:
            print(f'ERROR fetching {team_url}:', e, file=sys.stderr)
        time.sleep(15)
    return False


def main():
    # find *_link.csv files in data dir
    patterns = glob.glob(os.path.join(DATA_DIR, '*_link.csv'))
    if not patterns:
        print('No *_link.csv files found in data/; run main.py to create them first.', file=sys.stderr)
        return

    # sort to process chronologically by filename (assumes filenames like 2010-2011_link.csv)
    patterns = sorted(patterns)
    for path in patterns:
        fname = os.path.basename(path)
        try:
            season = fname.split('_')[0]
            # extract beginning year
            year = int(season.split('-')[0])
        except Exception:
            print('Skipping unknown file', fname)
            continue
        if year < START_YEAR:
            continue

        print('\nProcessing season', season, 'from', path)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print('ERROR reading', path, e, file=sys.stderr)
            continue

        # ensure squad_href column exists
        if 'squad_href' not in df.columns:
            print('No squad_href column in', path, '- skipping', file=sys.stderr)
            continue

        for idx, row in df.iterrows():
            squad = row.get('Squad') or row.get('squad') or str(row.get(df.columns[1] if len(df.columns) > 1 else ''))
            href = row.get('squad_href', '')
            if not isinstance(href, str) or not href.strip():
                print('Skipping', squad, '- no href')
                continue
            href = href.strip()
            # ensure absolute
            if href.startswith('/'):
                href = urljoin('https://fbref.com', href)

            success = extract_and_save(href, season, squad)
            if not success:
                print('Failed to extract for', squad, '->', href, file=sys.stderr)
            time.sleep(5)


if __name__ == '__main__':
    main()
