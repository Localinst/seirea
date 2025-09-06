import json
from pathlib import Path
import pandas as pd
import numpy as np
import importlib.util


def load_compute_features(path_to_build):
    spec = importlib.util.spec_from_file_location('build_features', str(path_to_build))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_features


def main():
    root = Path(__file__).resolve().parents[1]
    combined = root / 'data' / 'combined_matches.csv'
    saved = root / 'data' / 'features.csv'
    build_py = root / 'features' / 'build_features.py'
    out_dir = root / 'artifacts_regression'
    out_dir.mkdir(parents=True, exist_ok=True)

    compute_features = load_compute_features(build_py)

    df_matches = pd.read_csv(combined, parse_dates=['Date'])
    recomputed = compute_features(df_matches)

    df_saved = pd.read_csv(saved, parse_dates=['Date'])

    # Merge on Date, HomeTeam, AwayTeam to align rows
    key = ['Date', 'HomeTeam', 'AwayTeam']
    merged = recomputed.merge(df_saved, on=key, how='outer', indicator=True, suffixes=('_rec', '_sav'))

    report = {'total_rows_recomputed': len(recomputed), 'total_rows_saved': len(df_saved), 'merged_rows': len(merged), 'mismatches': {}}

    # find numeric columns present in both
    rec_cols = [c for c in recomputed.columns if c not in key]
    sav_cols = [c for c in df_saved.columns if c not in key]
    common = set(rec_cols) & set(sav_cols)

    tol = 1e-6
    sample_limit = 5

    for col in sorted(common):
        col_rec = col + '_rec'
        col_sav = col + '_sav'
        if col_rec not in merged.columns or col_sav not in merged.columns:
            continue
        a = merged[col_rec].to_numpy()
        b = merged[col_sav].to_numpy()
        # consider NaNs equal
        # handle numeric vs non-numeric safely
        neq = None
        try:
            # numeric comparison
            neq = ~((pd.isna(a) & pd.isna(b)) | ((~pd.isna(a)) & (~pd.isna(b)) & (np.isclose(a.astype(float), b.astype(float), atol=tol, equal_nan=True))))
        except Exception:
            # fallback to string equality
            neq = ~((pd.isna(a) & pd.isna(b)) | ((~pd.isna(a)) & (~pd.isna(b)) & (a == b)))
        neq_count = int(neq.sum())
        if neq_count > 0:
            idxs = list(np.where(neq)[0][:sample_limit])
            examples = []
            for i in idxs:
                examples.append({'Date': str(merged.iloc[i]['Date']), 'HomeTeam': merged.iloc[i]['HomeTeam'], 'AwayTeam': merged.iloc[i]['AwayTeam'], 'recomputed': merged.iloc[i][col_rec], 'saved': merged.iloc[i][col_sav]})
            report['mismatches'][col] = {'count': neq_count, 'examples': examples}

    out_file = out_dir / 'validate_features_report.json'
    with open(out_file, 'w', encoding='utf8') as f:
        json.dump(report, f, indent=2, default=str)

    print('Validation complete. Report:', out_file)


if __name__ == '__main__':
    main()
