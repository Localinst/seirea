"""Concatena file *_standard.csv e *_shooting.csv in file unici aggiungendo
la prima colonna 'stagione' estratta dal nome del file.

Uso:
  python concat_standard_shooting.py [--dir data] [--dry-run] [--no-backup] [--verbose]

Genera per default:
 - data/combined_standard.csv
 - data/combined_shooting.csv
"""
from pathlib import Path
import argparse
import csv
import sys
import shutil
from datetime import datetime


def detect_encoding(path: Path):
    try:
        path.read_text(encoding='utf-8')
        return 'utf-8'
    except Exception:
        return 'latin-1'


def extract_season_from_name(path: Path):
    stem = path.stem
    parts = stem.split('_')
    return parts[0] if parts else stem


def find_files(data_dir: Path, pattern: str):
    return sorted(data_dir.glob(pattern))


def concat_pattern(data_dir: Path, pattern: str, output: Path, dry_run=True, backup=True, verbose=False):
    files = find_files(data_dir, pattern)
    if not files:
        if verbose:
            print(f"Nessun file {pattern} trovato in {data_dir}")
        return False

    total_rows = 0
    summaries = []

    if dry_run:
        for f in files:
            enc = detect_encoding(f)
            with f.open('r', encoding=enc, newline='') as fh:
                reader = csv.reader(fh)
                try:
                    header = next(reader)
                except StopIteration:
                    rows = 0
                else:
                    rows = sum(1 for _ in reader)
            season = extract_season_from_name(f)
            summaries.append((f.name, season, len(header) if 'header' in locals() else 0, rows))
            total_rows += rows

        print(f"Dry-run for pattern '{pattern}':")
        for name, season, ncols, rows in summaries:
            print(f" - {name}: stagione={season}, colonne={ncols}, righe={rows}")
        print(f"Totale righe concatenate per pattern '{pattern}': {total_rows}\n")
        return True

    # backup existing output
    if output.exists() and backup:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = output.with_name(output.name + f'.bak-{ts}')
        shutil.copy2(output, backup_path)
        if verbose:
            print(f"[BACKUP] {output} -> {backup_path}")

    written = 0
    header_written = False
    with output.open('w', encoding='utf-8', newline='') as outfh:
        writer = None
        for f in files:
            enc = detect_encoding(f)
            season = extract_season_from_name(f)
            with f.open('r', encoding=enc, newline='') as fh:
                reader = csv.reader(fh)
                try:
                    hdr = next(reader)
                except StopIteration:
                    if verbose:
                        print(f"[SKIP] {f} è vuoto")
                    continue

                if not header_written:
                    out_header = ['stagione'] + hdr
                    writer = csv.writer(outfh)
                    writer.writerow(out_header)
                    header_written = True

                for row in reader:
                    writer.writerow([season] + row)
                    written += 1

    if verbose:
        print(f"[OK] Scritti {written} righe in {output}")
    else:
        print(f"Scritti {written} righe in {output}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description='Concatena *_standard.csv e *_shooting.csv aggiungendo la colonna stagione')
    parser.add_argument('--dir', '-d', default='data', help='Cartella contenente i CSV')
    parser.add_argument('--dry-run', action='store_true', help='Mostra le azioni senza scrivere')
    parser.add_argument('--no-backup', action='store_true', help="Non fare backup dell'output esistente")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose')
    args = parser.parse_args(argv)

    data_dir = Path(args.dir)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Cartella non trovata: {data_dir}")
        sys.exit(2)

    ok1 = concat_pattern(data_dir, '*_standard.csv', data_dir / 'combined_standard.csv', dry_run=args.dry_run, backup=not args.no_backup, verbose=args.verbose)
    ok2 = concat_pattern(data_dir, '*_shooting.csv', data_dir / 'combined_shooting.csv', dry_run=args.dry_run, backup=not args.no_backup, verbose=args.verbose)

    if not ok1 and not ok2:
        print('Nessun file concatenato')
        sys.exit(1)


if __name__ == '__main__':
    main()
