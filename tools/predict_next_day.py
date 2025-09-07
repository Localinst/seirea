#!/usr/bin/env python3
"""
Read 'prossima_giornata.csv' (Home, Away, date) and run tools/predict_with_bookies.py for each row.
Produce 'predizioni_prossima_giornata.csv' with columns:
 Home, Away, Prediction (H/D/A), probability, quota-equa, quota-minima-accettare

Usage:
 python tools/predict_next_day.py --input prossima_giornata.csv --output predizioni_prossima_giornata.csv

This script calls the existing `tools/predict_with_bookies.py` script for each match and
parses the JSON printed to stdout (fallbacks to artifact JSON if stdout isn't JSON).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from typing import Optional


def safe_fname(s: str) -> str:
    # make a filesystem-safe filename fragment
    return re.sub(r"[^0-9A-Za-z._-]", "_", s)


def run_predict(home: str, away: str, date: str, repo_root: str) -> Optional[dict]:
    """Call predict_with_bookies.py and return parsed JSON dict or None on failure."""
    script = os.path.join(repo_root, "tools", "predict_with_bookies.py")
    cmd = [sys.executable, script, "--home", home, "--away", away, "--date", date]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        print(f"Error running predict script: {e}")
        return None

    stdout = proc.stdout.strip()
    # try parse stdout as JSON
    if stdout:
        try:
            return json.loads(stdout)
        except Exception:
            # not JSON, continue to fallback
            pass

    # fallback: check artifact file
    art_name = f"single_match_bookie_check_{safe_fname(home)}_{safe_fname(away)}_{safe_fname(date)}.json"
    art_path = os.path.join(repo_root, "artifacts", art_name)
    if os.path.exists(art_path):
        try:
            with open(art_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    # If exact artifact not found, try a permissive search in artifacts for files
    # that include the home team and date — this handles typos in the Away name
    # (e.g., CSV has 'Iner' but artifact used 'Inter').
    art_dir = os.path.join(repo_root, "artifacts")
    if os.path.isdir(art_dir):
        try:
            for fn in os.listdir(art_dir):
                if not fn.startswith("single_match_bookie_check_") or not fn.endswith('.json'):
                    continue
                lower = fn.lower()
                if safe_fname(home).lower() in lower and safe_fname(date).lower() in lower:
                    candidate = os.path.join(art_dir, fn)
                    try:
                        with open(candidate, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception:
                        continue
        except Exception:
            pass

    # last resort: try to parse any JSON object in stdout
    try:
        idx = stdout.find('{')
        if idx != -1:
            return json.loads(stdout[idx:])
    except Exception:
        return None

    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="prossima_giornata.csv", help="input CSV with Home,Away,date")
    p.add_argument("--output", default="predizioni_prossima_giornata.csv", help="output CSV")
    p.add_argument("--threshold", type=float, default=1.25, help="threshold multiplier for min acceptable odds")
    p.add_argument("--repo-root", default=os.getcwd(), help="repository root path (defaults to CWD)")
    args = p.parse_args(argv)

    in_path = args.input
    out_path = args.output
    threshold = float(args.threshold)
    repo_root = args.repo_root

    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}")
        return 2

    rows = []
    with open(in_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            # accept different column name casings
            home = r.get('Home') or r.get('home') or r.get('HOME') or r.get('HomeTeam')
            away = r.get('Away') or r.get('away') or r.get('AWAY') or r.get('AwayTeam')
            date = r.get('date') or r.get('Date') or r.get('DATA')
            if not (home and away and date):
                print(f"Skipping row {i+1}: missing Home/Away/date -> {r}")
                continue

            print(f"Predicting: {home} vs {away} on {date}")
            result = run_predict(home, away, date, repo_root)
            if result is None:
                print(f"Warning: no prediction result for {home} - {away} ({date})")
                continue

            # find probability for predicted outcome
            # the predict script returns keys: 'predicted' (H/D/A) and 'predicted_pct' or 'predicted_prob'
            pred = result.get('predicted') or result.get('Prediction') or result.get('predicted_outcome')
            prob = result.get('predicted_pct') or result.get('predicted_prob') or result.get('probability')
            try:
                prob = float(prob)
            except Exception:
                prob = None

            if pred is None or prob is None:
                print(f"Incomplete result for {home}-{away}: {result}")
                continue

            fair = None
            try:
                fair = 1.0 / prob if prob > 0 else None
            except Exception:
                fair = None

            min_acc = fair * threshold if fair is not None else None

            rows.append({
                'Home': home,
                'Away': away,
                'Prediction': pred,
                'probability': f"{prob:.6f}",
                'quota-equa': f"{fair:.6f}" if fair is not None else '',
                'quota-minima-accettare': f"{min_acc:.6f}" if min_acc is not None else ''
            })

    # write output
    if rows:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Home', 'Away', 'Prediction', 'probability', 'quota-equa', 'quota-minima-accettare']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote {len(rows)} predictions to {out_path}")
        return 0
    else:
        print("No predictions produced.")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
