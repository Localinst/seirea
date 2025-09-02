"""Rimuove la prima riga da tutti i file in `data/` che finiscono con
"*_results.csv" o "*_shooting.csv".

Uso:
  python pulizia.py [--dir DATA_DIR] [--dry-run] [--no-backup] [--verbose]

Comportamento:
 - Per default crea una copia di backup per ogni file processato.
 - Con --dry-run mostra le azioni senza modificare i file.
"""
from pathlib import Path
import argparse
import shutil
import sys
from datetime import datetime


def process_file(path: Path, backup: bool = True, dry_run: bool = True, verbose: bool = False) -> bool:
	"""Rimuove la prima riga dal file indicato.

	Returns True se il file sarebbe/stato modificato, False altrimenti.
	"""
	try:
		# Proviamo prima con utf-8, poi con latin-1 se fallisce
		try:
			text = path.read_text(encoding="utf-8")
			enc = "utf-8"
		except Exception:
			text = path.read_text(encoding="latin-1")
			enc = "latin-1"
	except Exception as e:
		if verbose:
			print(f"[ERROR] Impossibile leggere {path}: {e}")
		return False

	if not text:
		if verbose:
			print(f"[SKIP] {path} è vuoto")
		return False

	lines = text.splitlines(keepends=True)
	if len(lines) <= 1:
		if verbose:
			print(f"[SKIP] {path} contiene {len(lines)} riga(e), niente da rimuovere")
		return False

	if dry_run:
		print(f"[DRY-RUN] Rimuoverei la prima riga da: {path} (righe: {len(lines)}) encoding={enc}")
		return True

	# backup
	if backup:
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		backup_path = path.with_name(path.name + f".bak-{ts}")
		shutil.copy2(path, backup_path)
		if verbose:
			print(f"[BACKUP] {path} -> {backup_path}")

	# Riscriviamo il file senza la prima riga
	try:
		with path.open("w", encoding=enc, newline="") as f:
			f.writelines(lines[1:])
		if verbose:
			print(f"[OK] Prima riga rimossa da: {path}")
		return True
	except Exception as e:
		print(f"[ERROR] Impossibile scrivere {path}: {e}")
		return False


def find_targets(data_dir: Path):
	patterns = ("*_resultss.csv", "*_shootingg.csv")
	files = []
	for p in patterns:
		files.extend(sorted(data_dir.glob(p)))
	return files


def main(argv=None):
	parser = argparse.ArgumentParser(description="Rimuove la prima riga dai CSV results/shooting in data/")
	parser.add_argument("--dir", "-d", default="data", help="Cartella contenente i CSV (default: data)")
	parser.add_argument("--dry-run", action="store_true", help="Mostra le azioni senza modificare i file")
	parser.add_argument("--no-backup", action="store_true", help="Non creare backup dei file originali")
	parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
	args = parser.parse_args(argv)

	data_dir = Path(args.dir)
	if not data_dir.exists() or not data_dir.is_dir():
		print(f"Cartella non trovata: {data_dir}")
		sys.exit(2)

	targets = find_targets(data_dir)
	if not targets:
		print(f"Nessun file matching in {data_dir}")
		return

	modified = 0
	skipped = 0
	for f in targets:
		ok = process_file(f, backup=not args.no_backup, dry_run=args.dry_run, verbose=args.verbose)
		if ok:
			modified += 1
		else:
			skipped += 1

	mode = "dry-run" if args.dry_run else "apply"
	print(f"\nRisultato: mode={mode}, target_files={len(targets)}, modified_or_would_modify={modified}, skipped={skipped}")


if __name__ == "__main__":
	main()

