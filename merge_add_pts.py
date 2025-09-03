import os
import pandas as pd

root = r"c:\Users\ReadyToUse\Desktop\Data\serie a"
combined_path = os.path.join(root, "data", "combined_all.csv")
dataset_path = os.path.join(root, "dataset_cleaned_for_ml.csv")

backup_path = os.path.join(root, "dataset_cleaned_for_ml_backup.csv")
out_path = os.path.join(root, "dataset_cleaned_for_ml_with_pts.csv")

print(f"Reading combined: {combined_path}")
print(f"Reading dataset: {dataset_path}")

combined = pd.read_csv(combined_path, dtype=str)
dataset = pd.read_csv(dataset_path, dtype=str)

# Normalize column names to english keys
combined_cols = {c: c for c in combined.columns}
if 'stagione' in combined.columns:
    combined_cols['stagione'] = 'season'
if 'Squad' in combined.columns:
    combined_cols['Squad'] = 'squad'
combined = combined.rename(columns=combined_cols)

# Normalize dataset header if it contains italian header row as first data row
# Drop any accidental duplicate header rows where season equals 'stagione' or 'season'
if dataset.shape[0] > 0:
    first = str(dataset.iloc[0, 0])
    if first.lower() in ['stagione', 'season']:
        print("Dropping an extra header-like first row in dataset_cleaned_for_ml.csv")
        dataset = dataset.iloc[1:].reset_index(drop=True)

# Ensure expected columns exist
if 'season' not in dataset.columns and 'stagione' in dataset.columns:
    dataset = dataset.rename(columns={'stagione': 'season'})
if 'squad' not in dataset.columns and 'Squad' in dataset.columns:
    dataset = dataset.rename(columns={'Squad': 'squad'})

# Trim and lower for matching
combined['season'] = combined['season'].astype(str).str.strip()
combined['squad'] = combined['squad'].astype(str).str.strip()
if 'Pts' in combined.columns:
    combined['Pts'] = pd.to_numeric(combined['Pts'], errors='coerce')
else:
    print('Warning: combined_all.csv has no Pts column')

dataset['season'] = dataset['season'].astype(str).str.strip()
dataset['squad'] = dataset['squad'].astype(str).str.strip()

# Lowercase squads for safer join, but keep original for output
dataset['_squad_lc'] = dataset['squad'].str.lower()
combined['_squad_lc'] = combined['squad'].str.lower()

# Merge on season and squad (lowercase)
merged = pd.merge(dataset, combined[['season','_squad_lc','Pts']], left_on=['season','_squad_lc'], right_on=['season','_squad_lc'], how='left')

# Report missing matches
missing = merged['Pts'].isna().sum()
print(f"Merged rows: {len(merged)}; missing Pts: {missing}")
if missing > 0:
    # show some examples
    miss_examples = merged[merged['Pts'].isna()][['season','squad']].drop_duplicates().head(10)
    print("Examples missing Pts:")
    print(miss_examples.to_string(index=False))

# Drop helper column and restore order
merged = merged.drop(columns=['_squad_lc'])

# Backup original dataset file
if not os.path.exists(backup_path):
    print(f"Backing up original dataset to {backup_path}")
    dataset.to_csv(backup_path, index=False)
else:
    print(f"Backup already exists at {backup_path}")

# Save merged output
merged.to_csv(out_path, index=False)
print(f"Saved merged dataset to {out_path}")

# Summary
print('\nSummary:')
print(f"Input dataset rows: {len(dataset)}")
print(f"Combined rows: {len(combined)}")
print(f"Output rows: {len(merged)}")
print('Done')
