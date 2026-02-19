"""
Data summary script: per-file and global SCADA stats.
Run: python -m src.data_summary [--max-minutes N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureConfig, get_running_and_buffer_blocks
from .load_data import build_base_frame, read_scada_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="SCADA data summary (per-file and merged).")
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=None,
        help="Cap merged timeline to this many minutes (for faster run on large data)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "Data"

    print("=" * 60)
    print("Per-file summary (signal CSVs)")
    print("=" * 60)

    for path in sorted(data_dir.glob("felsov02_*.csv")):
        if path.name.endswith("_sg.csv"):
            continue
        try:
            df = read_scada_csv(path)
            n = len(df)
            if "Date" in df.columns and n > 0:
                dmin = df["Date"].min()
                dmax = df["Date"].max()
                print(f"  {path.name}: {n:,} rows, Date {dmin} to {dmax}")
            else:
                print(f"  {path.name}: {n:,} rows")
        except Exception as e:
            print(f"  {path.name}: ERROR - {e}")

    comm_path = data_dir / "felsov02_sg.csv"
    if comm_path.exists():
        try:
            df = read_scada_csv(comm_path)
            n = len(df)
            if "Date" in df.columns and n > 0:
                dmin = df["Date"].min()
                dmax = df["Date"].max()
                print(f"  {comm_path.name}: {n:,} rows, Date {dmin} to {dmax}")
            else:
                print(f"  {comm_path.name}: {n:,} rows")
        except Exception as e:
            print(f"  {comm_path.name}: ERROR - {e}")

    print()
    print("=" * 60)
    print("Merged base frame" + (" (max_minutes={})".format(args.max_minutes) if args.max_minutes else " (full timeline)"))
    print("=" * 60)

    try:
        df = build_base_frame(data_dir, max_minutes=args.max_minutes)
        total = len(df)
        dmin = df.index.min()
        dmax = df.index.max()
        print(f"  Global date range: {dmin} to {dmax}")
        print(f"  Total rows (1-min reindex): {total:,}")

        data_cols = [c for c in df.columns if c not in ("Comm_hi", "expected_data", "missing_expected")]
        has_any = (~df[data_cols].isna().all(axis=1)).sum()
        print(f"  Rows with at least one non-NaN signal: {has_any:,}")

        if "missing_expected" in df.columns:
            missing = df["missing_expected"].sum()
            print(f"  Missing expected data: {missing:,}")

        if "Gen_cb_cld" in df.columns:
            s = df["Gen_cb_cld"]
            running = pd.Series(
                np.where(s.isna(), False, s.astype(bool)), index=df.index, dtype=bool
            )
            run_minutes = running.sum()
            print(f"  Running minutes (Gen_cb_cld True): {run_minutes:,}")

            config = FeatureConfig()
            buffer_minutes = max(config.analog_windows) if config.analog_windows else 60
            blocks = get_running_and_buffer_blocks(
                df,
                run_col="Gen_cb_cld",
                min_run_minutes=config.min_run_minutes,
                warmup_minutes=config.warmup_minutes,
                buffer_minutes=buffer_minutes,
            )
            n_subset = sum(len(df.loc[start_ts:end_ts]) for start_ts, end_ts in blocks)
            print(f"  Running+buffer blocks: {len(blocks)}")
            print(f"  Rows in running+buffer subset: {n_subset:,} (used for features)")
    except Exception as e:
        print(f"  ERROR: {e}")
        raise

    print()
    print("Done. Use start_date/end_date or max_minutes in build_base_frame for smaller runs.")


if __name__ == "__main__":
    main()

