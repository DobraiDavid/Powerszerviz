from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd


def _parse_datetime_column(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Ensure the given date column is parsed as pandas datetime and sorted.
    Drops exact duplicate timestamps, keeping the last occurrence.
    """
    if date_col not in df.columns:
        raise ValueError(f"Expected datetime column '{date_col}' not found in dataframe.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Keep the last row when there are duplicate timestamps in the same file
    df = df.drop_duplicates(subset=[date_col], keep="last")
    return df


def _coerce_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns that only contain True/False (or 0/1) values to boolean dtype.
    This helps later when we want to distinguish between analog and binary signals.
    """
    df = df.copy()
    for col in df.columns:
        if col == "Date":
            continue

        series = df[col]
        # Skip non-object/numeric quickly
        if series.dtype == "bool":
            continue

        # Try to interpret as booleans
        unique_vals = set(series.dropna().unique().tolist())
        if unique_vals.issubset({0, 1}) or unique_vals.issubset({True, False}):
            df[col] = series.astype(bool)
        elif unique_vals.issubset({"0", "1"}):
            df[col] = series.astype(int).astype(bool)
        elif unique_vals.issubset({"True", "False"}):
            df[col] = series.map({"True": True, "False": False})

    return df


def read_scada_csv(path: Path) -> pd.DataFrame:
    """
    Read a single SCADA CSV file (felsov02_xx.csv) into a cleaned dataframe.

    - Parses the 'Date' column as datetime.
    - Sorts by time and removes duplicate timestamps within this file.
    - Attempts to infer boolean columns.
    """
    df = pd.read_csv(path)
    df = _parse_datetime_column(df, "Date")
    df = _coerce_bool_columns(df)
    return df


def load_signal_tables(
    data_dir: Path,
    include_patterns: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all SCADA signal tables (felsov02_01.csv, felsov02_02.csv, ...) from data_dir.

    Returns a dict mapping table name (stem) -> dataframe.
    The special communication helper table 'felsov02_sg.csv' is not included here.
    """
    data_dir = Path(data_dir)

    if include_patterns is None:
        # Default: all felsov02_0X and felsov02_XX style files except *_sg
        include_patterns = ["felsov02_*.csv"]

    tables: Dict[str, pd.DataFrame] = {}

    for pattern in include_patterns:
        for path in sorted(data_dir.glob(pattern)):
            if path.name.endswith("_sg.csv"):
                continue

            df = read_scada_csv(path)
            tables[path.stem] = df

    if not tables:
        raise FileNotFoundError(
            f"No SCADA CSV tables found in {data_dir} matching patterns {list(include_patterns)}"
        )

    return tables


def load_comm_table(data_dir: Path, filename: str = "felsov02_sg.csv") -> pd.DataFrame:
    """
    Load the communication helper table (felsov02_sg.csv).

    This table contains:
      - Date      : timestamp
      - Packtime  : internal counter
      - Comm_hi   : boolean flag indicating communication problem
    """
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Communication table not found at {path}")

    df = pd.read_csv(path)
    df = _parse_datetime_column(df, "Date")
    df = _coerce_bool_columns(df)
    return df


def build_base_frame(
    data_dir: Path,
    *,
    include_patterns: Optional[Iterable[str]] = None,
    minute_freq: str = "1min",
    start_date: Optional[Union[pd.Timestamp, str]] = None,
    end_date: Optional[Union[pd.Timestamp, str]] = None,
    max_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a single, time-indexed dataframe by horizontally merging all signal tables
    and aligning them on a continuous per-minute timeline.

    Steps:
      1) Load and clean all felsov02_xx.csv tables (except _sg).
      2) Outer-join them on ['Date', 'Packtime'] when available, otherwise on 'Date'.
      3) Set 'Date' as index, sort, and reindex to a continuous 1-minute index.
      4) Optionally restrict the timeline to start_date/end_date and/or max_minutes.
      5) Join the communication table and mark minutes where data was "expected"
         but missing (based on Comm_hi).

    The returned dataframe has:
      - DatetimeIndex at 1-minute frequency.
      - All signal columns from the source tables.
      - Columns:
          * 'Packtime'         : from the first table that had it (if present).
          * 'Comm_hi'          : from felsov02_sg.csv.
          * 'expected_data'    : True if we expect data in that minute (Comm_hi == False).
          * 'missing_expected' : True where signals are missing although expected_data is True.

    Optional limits (for faster runs on a subset of history):
      - start_date, end_date: only include minutes in this closed interval.
      - max_minutes: cap the length of the timeline (e.g. 43200 for 30 days).
    """
    data_dir = Path(data_dir)

    tables = load_signal_tables(data_dir, include_patterns=include_patterns)

    # Merge all tables on Date (and Packtime when present in both)
    merged: Optional[pd.DataFrame] = None
    for name, df in tables.items():
        join_cols = ["Date"]
        if "Packtime" in df.columns:
            join_cols.append("Packtime")

        if merged is None:
            merged = df
        else:
            on_cols = [c for c in join_cols if c in merged.columns]
            merged = pd.merge(
                merged,
                df,
                on=on_cols,
                how="outer",
                suffixes=("", f"_{name}"),
            )

    if merged is None:
        raise RuntimeError("No tables were merged – this should not happen.")

    # Ensure datetime index with continuous 1-minute steps
    merged = _parse_datetime_column(merged, "Date")
    merged = merged.set_index("Date").sort_index()

    full_index = pd.date_range(
        start=merged.index.min(),
        end=merged.index.max(),
        freq=minute_freq,
    )

    # Optional: restrict timeline to reduce rows
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        full_index = full_index[full_index >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        full_index = full_index[full_index <= end_ts]
    if max_minutes is not None and len(full_index) > max_minutes:
        full_index = full_index[:max_minutes]

    merged = merged.reindex(full_index)
    merged.index.name = "Date"

    # Join communication table
    try:
        comm = load_comm_table(data_dir)
        comm = comm.set_index("Date").sort_index()
        comm = comm.reindex(full_index)
        merged["Comm_hi"] = comm.get("Comm_hi")
    except FileNotFoundError:
        # If the communication helper table is missing, we simply skip it.
        merged["Comm_hi"] = pd.NA

    # expected_data = not communication error
    merged["expected_data"] = merged["Comm_hi"] == False  # noqa: E712

    # missing_expected: all-nan row while we expected data
    data_cols = [c for c in merged.columns if c not in ("Comm_hi", "expected_data")]
    row_all_nan = merged[data_cols].isna().all(axis=1)
    merged["missing_expected"] = row_all_nan & merged["expected_data"].fillna(False)

    return merged


__all__ = [
    "read_scada_csv",
    "load_signal_tables",
    "load_comm_table",
    "build_base_frame",
]

