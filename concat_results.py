"""Concatena tutti i file *_results.csv in una singola tabella aggiungendo
la prima colonna con la stagione estratta dal nome del file.

Esempio di nome file: 2010-2011_results.csv -> stagione = '2010-2011'

Uso:
  python concat_results.py [--dir data] [--output data/combined_results.csv] [--dry-run] [--no-backup] [--verbose]
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


def find_results_files(data_dir: Path):
    return sorted(data_dir.glob('*_resultss.csv'))


def extract_season_from_name(path: Path):
    # take the part before the first underscore
    stem = path.stem
    parts = stem.split('_')
    return parts[0] if parts else stem


def concat_files(data_dir: Path, output: Path, dry_run=True, backup=True, verbose=False):
    files = find_results_files(data_dir)
    if not files:
        print(f"Nessun file *_results.csv trovato in {data_dir}")
        return False

    total_rows = 0
    file_summaries = []

    # dry-run: just report counts
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
            file_summaries.append((f.name, season, len(header), rows))
            total_rows += rows

        print("Dry-run: file trovati e conteggi (righe senza intestazione):")
        for name, season, ncols, rows in file_summaries:
            print(f" - {name}: stagione={season}, colonne={ncols}, righe={rows}")
        print(f"Totale righe concatenate (senza header): {total_rows}")
        return True

    # non dry-run: write output
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

    print(f"Scritti {written} righe in {output} (header incluso)")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description='Concatena *_results.csv aggiungendo la colonna stagione')
    parser.add_argument('--dir', '-d', default='data', help='Cartella contenente i CSV')
    parser.add_argument('--output', '-o', default='data/combined_results.csv', help='File di output')
    parser.add_argument('--dry-run', action='store_true', help='Mostra le azioni senza scrivere')
    parser.add_argument('--no-backup', action='store_true', help='Non fare backup dell\'output esistente')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose')
    args = parser.parse_args(argv)

    data_dir = Path(args.dir)
    output = Path(args.output)

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Cartella non trovata: {data_dir}")
        sys.exit(2)

    ok = concat_files(data_dir, output, dry_run=args.dry_run, backup=not args.no_backup, verbose=args.verbose)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
