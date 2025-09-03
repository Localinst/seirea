import os
import time
import sys
import pandas as pd
import requests
from urllib.parse import urljoin

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Limit number of seasons processed (for safety/testing). Set env var MAX_SEASONS to override.
MAX_SEASONS = int(os.environ.get('MAX_SEASONS', '15'))

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

for x in range(MAX_SEASONS):
    anno1 = 2017 + x
    anno2 = 2018 + x
    season = f'{anno1}-{anno2}'
    url = f'https://fbref.com/en/comps/11/{season}/{season}-Serie-A-Stats'
    print('\nProcessing', season, '->', url)
    try:
        tables = pd.read_html(url)
    except Exception as e:
        print('SKIP', season, 'read_html failed:', e, file=sys.stderr)
        # wait a bit and continue
        time.sleep(5)
        continue

    
    results = None

    # Prefer parsing the HTML with BeautifulSoup so we can extract links in the 'squad' column
    page = None
    try:
        resp = requests.get(url, headers={'User-Agent': 'python-requests/0.0'})
        resp.raise_for_status()
        page = resp.text
        soup = BeautifulSoup(page, 'html.parser') if BeautifulSoup else None
    except Exception as e:
        print('SKIP', season, 'requests failed:', e, file=sys.stderr)
        time.sleep(1)
        continue

    # Try to find the exact results table by id (preferred) or fallback to heuristics
    table_id = f"results{anno1}-{anno2}111_overall"
    table_tag = None
    if soup:
        table_tag = soup.find('table', id=table_id)
    if table_tag is None:
        # fallback: try pandas heuristics on the full page
        for t in tables:
            try:
                if results is None and (has_substrs(t, ['Age']) or has_substrs(t, ['Squad', 'squad'])):
                    results = t
                    # we still want the soup table_tag if possible
            except Exception:
                continue
        # try to read by attrs if pandas didn't find
        if results is None:
            try:
                std_tables = pd.read_html(url, attrs={"id": table_id})
                if std_tables:
                    results = std_tables[0]
            except Exception:
                pass
    else:
        # convert HTML table tag to DataFrame using pandas
        try:
            results = pd.read_html(str(table_tag))[0]
        except Exception:
            results = None
   
    # Save found tables and attempt to extract squad links to scrape players
    if results is None:
        print('results table not found for', season, file=sys.stderr)
        time.sleep(1)
        continue

    # Try to extract squad hrefs from the HTML table and attach to the DataFrame
    squad_hrefs = []
    if soup and table_tag is not None:
        # find header index for squad column
        squad_idx = None
        thead = table_tag.find('thead')
        if thead:
            header_cells = thead.find_all('th')
            for idx, th in enumerate(header_cells):
                txt = th.get_text(strip=True).lower()
                if 'squad' in txt or 'team' in txt or 'club' in txt:
                    squad_idx = idx
                    break

        tbody = table_tag.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                if tr.get('class') and 'thead' in tr.get('class'):
                    continue
                cells = tr.find_all(['td', 'th'])
                target_cell = None
                if squad_idx is not None and squad_idx < len(cells):
                    target_cell = cells[squad_idx]
                else:
                    # fallback: anchor with '/en/squads/' first
                    for a in tr.find_all('a', href=True):
                        if '/en/squads/' in a['href']:
                            target_cell = a.parent
                            break
                    if target_cell is None:
                        a_first = tr.find('a', href=True)
                        if a_first:
                            target_cell = a_first.parent

                href = ''
                if target_cell:
                    a = target_cell.find('a', href=True)
                    if a:
                        href = urljoin('https://fbref.com', a['href'])
                squad_hrefs.append(href)

    # If we obtained hrefs and they match the DataFrame length, attach directly.
    if squad_hrefs and len(squad_hrefs) == len(results):
        results['squad_href'] = squad_hrefs
    else:
        # Fallback: try to map by squad name (case-insensitive)
        # build mapping from extracted anchors if available
        mapping = {}
        for h in squad_hrefs:
            # mapping by last part of href isn't reliable; use soup extraction above instead
            pass
        # find candidate squad column in results
        squad_col = None
        for c in results.columns:
            if isinstance(c, str) and any(k in c.lower() for k in ('squad','team','club')):
                squad_col = c
                break
        if squad_col is not None and squad_hrefs:
            # try a naive positional mapping for missing lengths: map by order until exhausted
            mapped = []
            for i, row in results.iterrows():
                if i < len(squad_hrefs):
                    mapped.append(squad_hrefs[i])
                else:
                    mapped.append('')
            results['squad_href'] = mapped
        else:
            # give an empty column so downstream code can rely on its presence
            results['squad_href'] = [''] * len(results)

    save_df(results, os.path.join(DATA_DIR, f'{season}_link.csv'))

    # Extract squad links from the HTML table rows (use soup if available for link extraction)
    squad_links = []
    if soup and table_tag is not None:
        # Find index of the 'Squad' column from the table header (thead)
        squad_idx = None
        thead = table_tag.find('thead')
        if thead:
            header_cells = thead.find_all('th')
            for idx, th in enumerate(header_cells):
                txt = th.get_text(strip=True).lower()
                if 'squad' in txt or 'team' in txt or 'club' in txt:
                    squad_idx = idx
                    break

        tbody = table_tag.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                # skip header/blank rows
                if tr.get('class') and 'thead' in tr.get('class'):
                    continue
                # get list of cells and pick the one at squad_idx if available
                cells = tr.find_all(['td', 'th'])
                target_cell = None
                if squad_idx is not None and squad_idx < len(cells):
                    target_cell = cells[squad_idx]
                else:
                    # fallback: prefer anchors that link to squad pages
                    for a in tr.find_all('a', href=True):
                        if '/en/squads/' in a['href']:
                            target_cell = a.parent
                            break
                    # last resort: use first anchor's parent
                    if target_cell is None:
                        a_first = tr.find('a', href=True)
                        if a_first:
                            target_cell = a_first.parent

                if target_cell:
                    a = target_cell.find('a', href=True)
                    if a:
                        squad_name = a.get_text(strip=True)
                        href = a['href']
                        team_url = urljoin('https://fbref.com', href)
                        squad_links.append((squad_name, team_url))
    else:
        # fallback: try to extract links by reading anchors from the full page
        if soup:
            # find all links in page that look like squads under comps/11
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True)
                if '/en/squads/' in href and text:
                    squad_links.append((text, urljoin('https://fbref.com', href)))

    # Deduplicate by squad name keeping order
    seen = set()
    uniq_squad_links = []
    for name, link in squad_links:
        if name.lower() not in seen:
            seen.add(name.lower())
            uniq_squad_links.append((name, link))

    # Helper to sanitize filenames
    def safe_name(s):
        return ''.join([c if c.isalnum() or c in (' ', '-', '_') else '_' for c in s]).strip().replace(' ', '_')

    # For each squad, fetch team page and extract the players table
   
   
    

    # polite delay
    time.sleep(1)
       
    