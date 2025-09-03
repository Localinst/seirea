"""Aggregate player-level CSVs into team-season features.

Searches for player CSVs under data/players or any file matching
"*_players.csv" or "*_stats_standard_11.csv" and computes team-level
aggregations per season and squad.

Outputs: data/team_aggregates.csv

Usage (PowerShell):
$env:START_YEAR=2010; python data\aggregate_players_to_team.py
"""
import os
import glob
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).resolve().parent
PLAYERS_GLOB = [
    str(DATA_DIR / '*stats_standard_11.csv'),
    str(DATA_DIR / '*_stats_standard_11.csv'),
]


def find_player_files():
    files = []
    # Prefer explicit players/<season>/*.csv discovery to avoid picking up link/results files
    players_root = DATA_DIR / 'players'
    if players_root.exists() and players_root.is_dir():
        for season_dir in sorted(players_root.iterdir()):
            if not season_dir.is_dir():
                continue
            for p in season_dir.glob('*stats_standard_11*.csv'):
                files.append(str(p))
            # also include any csv in the season folder just in case
            for p in season_dir.glob('*.csv'):
                if str(p).lower().endswith('.csv') and ('_link' not in p.name.lower()):
                    files.append(str(p))
    else:
        # fallback: previous behaviour (workspace root csvs)
        for pattern in PLAYERS_GLOB:
            files.extend(glob.glob(pattern))
        files.extend(glob.glob(str(DATA_DIR / '*.csv')))

    # unique and keep only csv files
    files = [f for f in sorted(set(files)) if f.lower().endswith('.csv')]

    # Filter out obvious non-player files by name (link/results) and validate by reading headers.
    candidate = []
    for f in files:
        ln = os.path.basename(f).lower()
        # skip link/result files and aggregate outputs
        if '_link' in ln or 'results' in ln or 'team_aggregates' in ln:
            continue
        # sniff raw file content to handle multi-row headers (fbref style)
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                raw = fh.read(4096)
        except Exception:
            continue
        raw_lc = raw.lower()
        # accept file if it contains likely player-level headers
        if any(tok in raw_lc for tok in ('player,', ',player', 'age,', ',age', 'g+a', 'gls', 'ast')):
            candidate.append(f)
    return candidate


def infer_season_squad_from_filename(path):
    name = os.path.basename(path)
    # try patterns like '2010-2011_Milan_players.csv' or '2010-2011_Milan_stats_standard_11.csv'
    parts = name.split('_')
    season = None
    squad = None
    # first try if first part looks like YYYY-YYYY
    if parts:
        if '-' in parts[0] and parts[0][0:4].isdigit():
            season = parts[0]
            if len(parts) >= 2:
                squad = parts[1]
        else:
            # maybe file like 'Milan_players_2010-2011.csv'
            for p in parts:
                if '-' in p and p[0:4].isdigit():
                    season = p
                elif squad is None and len(p) > 1 and not p.endswith('.csv'):
                    squad = p
    return season, squad


def safe_numeric(df, col):
    # case-insensitive lookup with substring fallback
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    lc = col.lower()
    for c in df.columns:
        if str(c).lower() == lc:
            return pd.to_numeric(df[c], errors='coerce')
    for c in df.columns:
        if lc in str(c).lower():
            return pd.to_numeric(df[c], errors='coerce')
    return pd.Series([np.nan] * len(df))


def read_player_file(path):
    """Read a player-level CSV robustly handling fbref multi-row headers.

    Returns a cleaned DataFrame or raises.
    """
    # try common header rows: 0, 1, or multiindex
    for header in (0, 1):
        try:
            df = pd.read_csv(path, header=header, encoding='utf-8', engine='python')
        except Exception:
            continue
        # flatten multiindex if any
        if isinstance(df.columns, pd.MultiIndex):
            cols = []
            for a, b in df.columns:
                name = b if (isinstance(b, str) and b.strip()) else a
                cols.append(str(name).strip())
            df.columns = cols
        else:
            df.columns = [str(c).strip() for c in df.columns]

        # if it looks like we found player columns, return
        cols_lc = [str(c).lower() for c in df.columns]
        if any(k in cols_lc for k in ('player','gls','ast','age')):
            # drop summary rows if present
            try:
                first_col = df.columns[0]
                df = df[~df[first_col].astype(str).str.contains('squad total|opponent total', case=False, na=False)]
            except Exception:
                pass
            # drop rows where Player cell is literally 'Player' (duplicate header rows)
            try:
                if 'player' in cols_lc:
                    player_col = df.columns[cols_lc.index('player')]
                    df = df[df[player_col].astype(str).str.strip().str.lower() != 'player']
            except Exception:
                pass
            return df

    # last-resort: read raw and attempt to parse second line as header
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = fh.readlines()
    if len(lines) >= 2:
        header_line = lines[1]
        try:
            df = pd.read_csv(path, header=0, names=[h.strip() for h in header_line.split(',')], skiprows=[0], engine='python')
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            pass
    # give up
    raise ValueError(f'Unable to parse player CSV {path}')


def aggregate_player_df(df, season_hint=None, squad_hint=None, starts_threshold=10):
    # normalize column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Try to get season and squad from columns if present
    season = None
    squad = None
    for c in df.columns:
        lc = c.lower()
        if 'season' == lc:
            season = df[c].iloc[0]
        if lc in ('squad','team','club'):
            squad = df[c].iloc[0]

    if season is None:
        season = season_hint
    if squad is None:
        squad = squad_hint

    # numeric columns we may use
    Gls = safe_numeric(df, 'Gls')
    Ast = safe_numeric(df, 'Ast')
    xG = safe_numeric(df, 'xG')
    Min = safe_numeric(df, 'Min')
    Sh = safe_numeric(df, 'Sh')
    Age = safe_numeric(df, 'Age')
    Starts = safe_numeric(df, 'Starts')
    CrdY = safe_numeric(df, 'CrdY')
    CrdR = safe_numeric(df, 'CrdR')
    PrgC = safe_numeric(df, 'PrgC')
    PrgP = safe_numeric(df, 'PrgP')
    PrgR = safe_numeric(df, 'PrgR')

    # per-player G+A fallback
    GplusA = None
    if 'G+A' in df.columns:
        GplusA = safe_numeric(df, 'G+A')
    else:
        GplusA = Gls.fillna(0) + Ast.fillna(0)

    total_players = len(df)
    starters = df[Starts >= 1] if 'Starts' in df.columns else df

    # role distribution
    pos = None
    if 'Pos' in df.columns:
        pos = df['Pos'].astype(str).fillna('').str.upper()
    elif 'Position' in df.columns:
        pos = df['Position'].astype(str).fillna('').str.upper()

    pct_gk = pct_df = pct_mf = pct_fw = np.nan
    if pos is not None:
        n = len(pos)
        pct_gk = (pos.str.contains('GK')).sum() / n if n else np.nan
        pct_df = (pos.str.contains('D')).sum() / n if n else np.nan
        pct_mf = (pos.str.contains('M')).sum() / n if n else np.nan
        pct_fw = (pos.str.contains('F')).sum() / n if n else np.nan

    # Aggregations
    agg = {
        'season': season,
        'squad': squad,
        'roster_size': total_players,
        'starters_count': int((Starts >= starts_threshold).sum()) if 'Starts' in df.columns else np.nan,
        'age_mean': float(Age.mean()) if not Age.dropna().empty else np.nan,
        'age_mean_starters': float(Age[Starts > starts_threshold].mean()) if ('Starts' in df.columns and not Age[Starts > starts_threshold].dropna().empty) else float(Age.mean()) if not Age.dropna().empty else np.nan,
        'pct_under23': float((Age < 23).sum() / total_players) if total_players else np.nan,
        'pct_gk': float(pct_gk) if not np.isnan(pct_gk) else np.nan,
        'pct_df': float(pct_df) if not np.isnan(pct_df) else np.nan,
        'pct_mf': float(pct_mf) if not np.isnan(pct_mf) else np.nan,
        'pct_fw': float(pct_fw) if not np.isnan(pct_fw) else np.nan,
        'goals_tot': float(Gls.sum()) if not Gls.dropna().empty else np.nan,
        'assists_tot': float(Ast.sum()) if not Ast.dropna().empty else np.nan,
        'xg_tot': float(xG.sum()) if not xG.dropna().empty else np.nan,
        'xga_proxy': np.nan,  # may be filled later
        'team_xg_per90': np.nan,
        'conversion_sh_to_g': float(Gls.sum()/Sh.sum()) if Sh.sum() > 0 else np.nan,
        'progressions_tot': float((PrgC.fillna(0) + PrgP.fillna(0) + PrgR.fillna(0)).sum()),
        'yellow_cards': float(CrdY.sum()) if not CrdY.dropna().empty else np.nan,
        'red_cards': float(CrdR.sum()) if not CrdR.dropna().empty else np.nan,
        'top5_mean_gpa': np.nan,
        'bottom5_mean_gpa': np.nan,
        'gap_top5_bottom5_gpa': np.nan,
        'age_std': float(Age.std()) if not Age.dropna().empty else np.nan,
    }

    # xGA proxy: if goalkeeper GA exists (column 'GA' per goalkeeper), sum it; else leave NaN
    if 'GA' in df.columns:
        GA = safe_numeric(df, 'GA')
        agg['xga_proxy'] = float(GA.sum()) if not GA.dropna().empty else np.nan
    else:
        agg['xga_proxy'] = np.nan

    # team xG per90 using minutes
    if not xG.dropna().empty and Min.sum() > 0:
        agg['team_xg_per90'] = float(xG.sum() / (Min.sum() / 90.0))

    # top5 / bottom5 on G+A
    gpa = GplusA.fillna(0)
    if len(gpa) >= 1:
        top5 = gpa.sort_values(ascending=False).head(5)
        bottom5 = gpa.sort_values(ascending=True).head(5)
        agg['top5_mean_gpa'] = float(top5.mean())
        agg['bottom5_mean_gpa'] = float(bottom5.mean())
        agg['gap_top5_bottom5_gpa'] = float(top5.mean() - bottom5.mean())

    return agg


def main(start_year=2010, out_file=None):
    files = find_player_files()
    print(f"Found {len(files)} candidate CSV files")
    # Group files by (season, squad) by reading and concatenating candidate files first
    grouped = {}
    for f in files:
        season_hint, squad_hint = infer_season_squad_from_filename(f)
        try:
            df = read_player_file(f)
        except Exception:
            # skip files we cannot parse robustly
            continue
        # attempt to extract season from df if missing
        if season_hint is None:
            for c in df.columns:
                if str(c).lower() == 'season':
                    season_hint = df[c].iloc[0]
                    break
        if season_hint is None:
            # try to infer season from parent directory name
            try:
                parent = Path(f).parent.name
                if '-' in parent and parent[0:4].isdigit():
                    season_hint = parent
            except Exception:
                pass
            if season_hint is None:
                continue
        try:
            sy = int(str(season_hint)[0:4])
        except Exception:
            try:
                sy = int(str(season_hint).split('-')[0])
            except Exception:
                continue
        if sy < start_year:
            continue

        # normalize squad
        if squad_hint is None:
            # try to find squad column
            for c in df.columns:
                if str(c).lower() in ('squad','team','club'):
                    try:
                        squad_hint = str(df[c].dropna().iloc[0])
                    except Exception:
                        squad_hint = str(df[c].iloc[0])
                    break
        key = (season_hint, squad_hint or '')
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(df)

    rows = []
    for (season_hint, squad_hint), df_list in grouped.items():
        # concatenate player rows for this team-season
        concat = pd.concat(df_list, ignore_index=True)
        # drop fully empty rows
        concat = concat.dropna(how='all')
        if concat.empty:
            continue
        agg = aggregate_player_df(concat, season_hint=season_hint, squad_hint=squad_hint)
        rows.append(agg)

    if not rows:
        print('No team aggregates produced (no player files found or none matched the start year).')
        return 1

    out = pd.DataFrame(rows)
    # Normalize season and squad strings
    out['season'] = out['season'].astype(str)
    out['squad'] = out['squad'].astype(str)

    # Read *_link.csv files to get canonical team lists per season and ensure 20 teams per season
    link_files = glob.glob(str(DATA_DIR / '*_link.csv'))
    for lf in sorted(link_files):
        try:
            link_df = pd.read_csv(lf)
        except Exception:
            continue
        # infer season from filename
        base = os.path.basename(lf)
        season_name = base.replace('_link.csv', '')
        # find squad column
        squad_col = None
        for c in link_df.columns:
            if 'squad' in str(c).lower() or 'team' in str(c).lower() or 'club' in str(c).lower():
                squad_col = c
                break
        if squad_col is None:
            # try first column as fallback
            squad_col = link_df.columns[1] if len(link_df.columns) > 1 else link_df.columns[0]
        squads = link_df[squad_col].astype(str).str.strip().tolist()
        # keep unique preserving order
        seen = set()
        squads_u = []
        for s in squads:
            if s.lower() not in seen and s.strip() != '':
                seen.add(s.lower())
                squads_u.append(s)
        # take first 20 if longer
        squads_u = squads_u[:20]

        for squad in squads_u:
            if not ((out['season'] == season_name) & (out['squad'].str.lower() == squad.lower())).any():
                # append empty aggregate row for missing team
                empty = {k: (np.nan if k not in ('season','squad') else None) for k in out.columns}
                empty['season'] = season_name
                empty['squad'] = squad
                # try to set roster_size if link_df has roster/MP column
                for c in link_df.columns:
                    if c.lower() in ('rk','mp','roster_size','players'):
                        try:
                            val = link_df[link_df[squad_col].astype(str).str.strip().str.lower() == squad.lower()][c].iloc[0]
                            empty['roster_size'] = int(val)
                        except Exception:
                            pass
                out = pd.concat([out, pd.DataFrame([empty])], ignore_index=True)

    # Deduplicate by season+squad keeping first found
    out['season'] = out['season'].astype(str)
    out['squad'] = out['squad'].astype(str)
    out = out.drop_duplicates(subset=['season','squad'], keep='first').reset_index(drop=True)
    out_file = Path(out_file) if out_file else (DATA_DIR / 'team_aggregates.csv')
    # write atomically to temporary file then move; if permission denied, try a fallback filename
    try:
        tmp = out_file.with_suffix('.tmp.csv')
        out.to_csv(tmp, index=False, encoding='utf-8-sig')
        try:
            tmp.replace(out_file)
        except Exception:
            # fallback to writing without replace
            out.to_csv(out_file, index=False, encoding='utf-8-sig')
        print('WROTE', out_file)
    except PermissionError:
        fallback = out_file.with_name(out_file.stem + '.new.csv')
        out.to_csv(fallback, index=False, encoding='utf-8-sig')
        print(f"Permission denied writing {out_file}; wrote fallback {fallback}")
    except Exception as e:
        print('ERROR writing aggregates:', e, file=sys.stderr)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=int(os.environ.get('START_YEAR', '2010')))
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    sys.exit(main(start_year=args.start_year, out_file=args.out))
