import pandas as pd
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    eval_path = root / "artifacts" / "simulation_evaluation.csv"
    i1_path = root / "partite" / "I1.csv"
    out_path = root / "artifacts" / "simulation_evaluation_ordered.csv"

    eval_df = pd.read_csv(eval_path, parse_dates=["Date"]) 
    eval_df["Date"] = pd.to_datetime(eval_df["Date"]).dt.date

    i1 = pd.read_csv(i1_path, dayfirst=True)
    i1["Date"] = pd.to_datetime(i1["Date"], dayfirst=True).dt.date
    try:
        i1["Time"] = pd.to_datetime(i1["Time"], format="%H:%M").dt.time
    except Exception:
        i1["Time"] = pd.to_datetime(i1["Time"], errors="coerce").dt.time

    # Join to get evaluation rows keyed as in I1
    merged = pd.merge(i1[["Date", "Time", "HomeTeam", "AwayTeam"]], eval_df, how="left", on=["Date", "HomeTeam", "AwayTeam"])

    # If eval already contains a Time column, prefer it (some rows may differ); otherwise use I1 Time
    if "Time_x" in merged.columns and "Time_y" in merged.columns:
        # Time_x from i1, Time_y from eval
        merged["Time"] = merged["Time_y"].fillna(merged["Time_x"])
        merged = merged.drop(columns=[c for c in merged.columns if c.startswith("Time_")])

    # Save ordered by Date, Time (Time may be NaT for some rows)
    sort_cols = ["Date"] + (["Time"] if "Time" in merged.columns else [])
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote ordered evaluation to {out_path}")

    # show first 10 rows for quick check
    print(merged[["Date", "Time", "HomeTeam", "AwayTeam", "predicted", "FTR", "correct"]].head(10))


if __name__ == '__main__':
    main()
