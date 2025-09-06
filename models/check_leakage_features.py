import json
import os
from pathlib import Path
import pandas as pd


def pts_from_result_for_team(result, is_home):
    # result: 'H','D','A'
    if result == 'D':
        return 1
    if is_home:
        return 3 if result == 'H' else 0
    else:
        return 3 if result == 'A' else 0


def analyze_season_aggregates(df):
    # We'll check these columns for home and away
    cols = [
        ('home', 'home_season_pts', 'home_season_wins', 'home_season_draws', 'home_season_losses', 'home_season_gf', 'home_season_ga', 'home_season_matches'),
        ('away', 'away_season_pts', 'away_season_wins', 'away_season_draws', 'away_season_losses', 'away_season_gf', 'away_season_ga', 'away_season_matches'),
    ]

    report = {"per_column": {}, "summary": {"total_rows": len(df)}}

    # Sort by Date then by original index to preserve order
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['_orig_idx'] = range(len(df))
    df = df.sort_values(['Date', '_orig_idx']).reset_index(drop=True)

    # Build per-team cumulative stats up to but excluding current match
    teams = {}

    # Initialize per-column counters
    counters = {}
    for side, *colnames in cols:
        for col in colnames:
            counters[col] = {"matches_equal_before": 0, "matches_equal_after": 0, "mismatch": 0}

    for i, row in df.iterrows():
        ftr = row.get('FTR')
        # If result missing, skip row for season-agg checks
        if pd.isna(ftr):
            continue

        for side, *colnames in cols:
            team = row['HomeTeam'] if side == 'home' else row['AwayTeam']
            is_home = side == 'home'
            if team not in teams:
                teams[team] = {
                    'pts': 0,
                    'wins': 0,
                    'draws': 0,
                    'losses': 0,
                    'gf': 0,
                    'ga': 0,
                    'matches': 0,
                }

            before = teams[team].copy()

            # Compare each requested column
            # determine what the value should be before and after
            # compute result-dependent values
            pts = pts_from_result_for_team(ftr, is_home)
            home_goals = int(row.get('FTHG')) if pd.notna(row.get('FTHG')) else None
            away_goals = int(row.get('FTAG')) if pd.notna(row.get('FTAG')) else None
            gf_scored = home_goals if is_home else away_goals
            ga_conceded = away_goals if is_home else home_goals

            after = {
                'pts': before['pts'] + (pts if pts is not None else 0),
                'wins': before['wins'] + (1 if (ftr == ('H' if is_home else 'A')) else 0),
                'draws': before['draws'] + (1 if ftr == 'D' else 0),
                'losses': before['losses'] + (1 if (ftr != 'D' and ftr != ('H' if is_home else 'A')) else 0),
                'gf': before['gf'] + (gf_scored if gf_scored is not None else 0),
                'ga': before['ga'] + (ga_conceded if ga_conceded is not None else 0),
                'matches': before['matches'] + 1,
            }

            # mapping of colnames to values
            mapping = {
                'home_season_pts': (before['pts'], after['pts']),
                'home_season_wins': (before['wins'], after['wins']),
                'home_season_draws': (before['draws'], after['draws']),
                'home_season_losses': (before['losses'], after['losses']),
                'home_season_gf': (before['gf'], after['gf']),
                'home_season_ga': (before['ga'], after['ga']),
                'home_season_matches': (before['matches'], after['matches']),
                'away_season_pts': (before['pts'], after['pts']),
                'away_season_wins': (before['wins'], after['wins']),
                'away_season_draws': (before['draws'], after['draws']),
                'away_season_losses': (before['losses'], after['losses']),
                'away_season_gf': (before['gf'], after['gf']),
                'away_season_ga': (before['ga'], after['ga']),
                'away_season_matches': (before['matches'], after['matches']),
            }

            for col in colnames:
                if col not in row:
                    continue
                val = row[col]
                # try numeric comparison
                try:
                    val_num = float(val) if pd.notna(val) else None
                except Exception:
                    val_num = None

                before_val, after_val = mapping.get(col, (None, None))

                matched_before = (val_num == before_val) if (val_num is not None and before_val is not None) else False
                matched_after = (val_num == after_val) if (val_num is not None and after_val is not None) else False

                if matched_before:
                    counters[col]['matches_equal_before'] += 1
                elif matched_after:
                    counters[col]['matches_equal_after'] += 1
                else:
                    counters[col]['mismatch'] += 1

            # Now update teams stats with current match (for next iterations)
            teams[team]['pts'] = after['pts']
            teams[team]['wins'] = after['wins']
            teams[team]['draws'] = after['draws']
            teams[team]['losses'] = after['losses']
            teams[team]['gf'] = after['gf']
            teams[team]['ga'] = after['ga']
            teams[team]['matches'] = after['matches']

    # Summarize
    for col, counts in counters.items():
        total = counts['matches_equal_before'] + counts['matches_equal_after'] + counts['mismatch']
        report['per_column'][col] = {
            'matches_equal_before': counts['matches_equal_before'],
            'matches_equal_after': counts['matches_equal_after'],
            'mismatch': counts['mismatch'],
            'rows_checked': total,
        }

    # simple heuristic: if majority equal_after -> likely leakage
    report['summary']['likely_leak_columns'] = [col for col, info in report['per_column'].items() if info['rows_checked']>0 and info['matches_equal_after'] > info['matches_equal_before']]
    report['summary']['likely_pre_match_columns'] = [col for col, info in report['per_column'].items() if info['rows_checked']>0 and info['matches_equal_before'] > info['matches_equal_after']]

    return report


def detect_obvious_postmatch_columns(df):
    # Columns that trivially leak because they are match events
    postmatch_names = ['FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    found = [c for c in postmatch_names if c in df.columns]
    return found


def main():
    cwd = Path(__file__).resolve().parents[1]
    csv_path = cwd / 'data' / 'features.csv'
    out_dir = cwd / 'artifacts_regression'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    report = {}
    report['obvious_postmatch_cols'] = detect_obvious_postmatch_columns(df)
    report['season_agg_analysis'] = analyze_season_aggregates(df)

    out_file = out_dir / 'leakage_report.json'
    with open(out_file, 'w', encoding='utf8') as f:
        json.dump(report, f, indent=2)

    print('Leakage check complete. Report saved to:', str(out_file))


if __name__ == '__main__':
    main()
