from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# Horizons in minutes
MAINT_HORIZON_1H_MIN = 1 * 60         # 60
MAINT_HORIZON_6H_MIN = 6 * 60         # 360
MAINT_HORIZON_24H_MIN = 24 * 60       # 1440
MAINT_HORIZON_3D_MIN = 3 * 24 * 60    # 4320
MAINT_HORIZON_7D_MIN = 7 * 24 * 60    # 10080 (used by add_maintenance_labels only)
MAINT_HORIZON_30D_MIN = 30 * 24 * 60  # 43200 (used by add_maintenance_labels only)

# Optional: if provided, require at least one of these True at stop (leállító hiba)
DEFAULT_SHUTDOWN_FAULT_COLUMNS = ["Leallas_zav", "Gen_trip"]

# Fault columns for "fault in future" labels. No complete mapping of error codes (1021, 1023, 1040, etc.) to
# SCADA columns exists; 452/453 doc is incomplete. We use Leallas_zav + Gen_trip to capture shutdown-type events.
DEFAULT_FAULT_COLUMNS: List[str] = ["Leallas_zav", "Gen_trip"]


def detect_run_stop_events(
    df: pd.DataFrame,
    *,
    run_col: str = "Gen_cb_cld",
) -> pd.Series:
    """Detect run-to-stop events: transition from running (True) to not running (False)."""
    if run_col not in df.columns:
        raise ValueError(f"Required run-state column '{run_col}' not found in dataframe.")

    s = df[run_col]
    running = pd.Series(np.where(s.isna(), False, s.astype(bool)), index=df.index, dtype=bool)
    shifted = running.shift(1)
    prev_running = pd.Series(np.where(shifted.isna(), False, shifted), index=df.index, dtype=bool)

    run_stop = (prev_running) & (~running)
    run_stop.name = "run_stop_event"
    return run_stop


def _downtime_minutes_after_stop(
    df: pd.DataFrame,
    run_stop: pd.Series,
    run_col: str = "Gen_cb_cld",
) -> pd.Series:
    """
    For each run_stop at t, compute how many minutes the motor stays off (until next run).
    Returns a series aligned with df.index: at run_stop minutes the value is downtime; elsewhere NaN.
    """
    running = pd.Series(
        np.where(df[run_col].isna(), False, df[run_col].astype(bool)),
        index=df.index,
        dtype=bool,
    )
    run_times = df.index[running].values.astype("datetime64[ns]")
    stop_times = df.index[run_stop].values.astype("datetime64[ns]")
    if len(run_times) == 0 or len(stop_times) == 0:
        return pd.Series(np.nan, index=df.index)

    idx_next = np.searchsorted(run_times, stop_times, side="right")
    valid = idx_next < len(run_times)
    # Only index run_times where valid (idx_next can equal len(run_times) when stop is after last run)
    next_run = np.empty(stop_times.shape, dtype=run_times.dtype)
    next_run[valid] = run_times[idx_next[valid]]
    next_run[~valid] = stop_times[~valid]  # dummy so delta is 0; we set downtime to nan for ~valid
    delta_ns = (next_run.astype("datetime64[ns]") - stop_times).astype("timedelta64[ns]")
    downtime_min = np.where(valid, delta_ns.astype(np.float64) / 60.0e9, np.nan)
    out = pd.Series(downtime_min, index=pd.Index(df.index[run_stop]))
    return out.reindex(df.index)


def add_maintenance_labels(
    df: pd.DataFrame,
    *,
    run_col: str = "Gen_cb_cld",
    oper_off_col: str = "Oper_off",
    shutdown_fault_columns: Optional[Iterable[str]] = None,
    min_downtime_minutes: int = 20,
    running_mask_col: str = "running_for_training",
) -> pd.DataFrame:
    """
    Add maintenance-need labels.

    Maintenance event = run→stop that:
    1) Is not voluntary (Oper_off is False at stop minute).
    2) At least one of shutdown_fault_columns True at stop (default: Leallas_zav, Gen_trip).
    3) Motor stayed off for at least min_downtime_minutes (default 20, filters brief glitches).

    By default shutdown_fault_columns is ["Leallas_zav", "Gen_trip"] and min_downtime_minutes is 20.
    Pass shutdown_fault_columns=[] to not require a fault signal at stop.

    Labels:
    - maintenance_event: as above.
    - y_maint_1h, y_maint_6h, y_maint_24h, y_maint_3d, y_maint_7d, y_maint_30d: 1 if any maintenance_event in next horizon (on running minutes).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have a DatetimeIndex before labeling.")

    df = df.copy()

    run_stop = detect_run_stop_events(df, run_col=run_col)

    # 1) Exclude voluntary: at stop minute Oper_off must be False
    if oper_off_col and oper_off_col in df.columns:
        s_off = df[oper_off_col]
        oper_off = pd.Series(
            np.where(s_off.isna(), False, s_off.astype(bool)), index=df.index, dtype=bool
        )
        fault_induced = run_stop & (~oper_off)
    else:
        fault_induced = run_stop.copy()

    # 2) Require at least one shutdown-fault column True at stop (default: Leallas_zav, Gen_trip)
    fault_cols = list(shutdown_fault_columns) if shutdown_fault_columns is not None else list(DEFAULT_SHUTDOWN_FAULT_COLUMNS)
    existing = [c for c in fault_cols if c in df.columns]
    # So that defaults are visible when this path is used:
    print(f"Adding maintenance labels (downtime >= {min_downtime_minutes} min, fault at stop: {existing or 'none'}).")
    if existing:
        # Avoid fillna(False).astype(bool) to prevent FutureWarning on object dtypes
        raw = df[existing].to_numpy(dtype=object, copy=False)
        vals = np.where(pd.isna(raw), False, raw)
        fault_at_stop = pd.Series(np.asarray(vals, dtype=bool).any(axis=1), index=df.index, dtype=bool)
        fault_induced = fault_induced & fault_at_stop
    # else: keep fault_induced as run_stop & ~oper_off

    # 3) Require minimum downtime after stop (motor off for at least N minutes)
    downtime = _downtime_minutes_after_stop(df, run_stop, run_col=run_col)
    at_stop = run_stop.reindex(df.index, fill_value=False).astype(bool)
    significant_downtime = at_stop & (downtime >= min_downtime_minutes)

    maintenance_event = fault_induced & significant_downtime
    maintenance_event.name = "maintenance_event"
    df["maintenance_event"] = maintenance_event

    s = df[run_col]
    running = pd.Series(np.where(s.isna(), False, s.astype(bool)), index=df.index, dtype=bool)
    df[running_mask_col] = running

    for horizon_min, label_col in [
        (MAINT_HORIZON_1H_MIN, "y_maint_1h"),
        (MAINT_HORIZON_6H_MIN, "y_maint_6h"),
        (MAINT_HORIZON_24H_MIN, "y_maint_24h"),
        (MAINT_HORIZON_3D_MIN, "y_maint_3d"),
        (MAINT_HORIZON_7D_MIN, "y_maint_7d"),
        (MAINT_HORIZON_30D_MIN, "y_maint_30d"),
    ]:
        label = make_horizon_label(maintenance_event, horizon_minutes=horizon_min)
        label = label.where(running, other=False)
        df[label_col] = label.astype(int)

    return df


def make_horizon_label(
    events: pd.Series,
    *,
    horizon_minutes: int,
) -> pd.Series:
    """Forward-looking binary label: 1 if any event in the next horizon_minutes, else 0."""
    if not isinstance(events.index, pd.DatetimeIndex):
        raise ValueError("events series must have a DatetimeIndex.")

    values = events.fillna(False).astype(int)
    window = int(horizon_minutes)
    rev = values.iloc[::-1].rolling(window=window, min_periods=1).max()
    label = rev.iloc[::-1].astype(bool)
    label.name = f"y_{horizon_minutes}"
    return label


def add_fault_horizon_labels(
    df: pd.DataFrame,
    *,
    fault_columns: Optional[Iterable[str]] = None,
    run_col: str = "Gen_cb_cld",
    running_mask_col: str = "running_for_training",
    max_ttf_minutes: int = 7 * 24 * 60,
) -> pd.DataFrame:
    """
    Add fault-horizon labels based on fault/trip signals (no run→stop or Oper_off).

    A "fault event" is any minute where at least one of fault_columns is True.
    A "fault episode" is the *first* minute of each contiguous run of fault events (one
    incident → one episode). Labels use episodes so one trip does not count as hundreds of events:
    y_fault_6h / y_fault_24h / y_fault_3d = 1 if any fault *episode* starts in next 6h / 24h / 3d.
    y_ttf = minutes until next episode start (capped at max_ttf_minutes). Both fault_event and
    fault_episode are added to the dataframe.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have a DatetimeIndex before labeling.")

    df = df.copy()
    cols = list(fault_columns) if fault_columns else list(DEFAULT_FAULT_COLUMNS)
    existing = [c for c in cols if c in df.columns]
    if not existing:
        raise ValueError(
            f"None of the fault columns {cols} found in dataframe. "
            "Provide fault_columns that exist (e.g. Leallas_zav, Gen_trip)."
        )

    raw = df[existing].to_numpy(dtype=object, copy=False)
    vals = np.where(pd.isna(raw), False, raw)
    fault_event = pd.Series(
        np.asarray(vals, dtype=bool).any(axis=1), index=df.index, dtype=bool
    )

    # Drop very short fault runs (< 2 minutes) – do not count them as faults at all.
    # This operates on contiguous True segments in fault_event.
    if fault_event.any():
        seg_id = (fault_event != fault_event.shift(1, fill_value=False)).cumsum()
        for s in seg_id[fault_event].unique():
            seg_mask = (seg_id == s) & fault_event
            idx_seg = df.index[seg_mask]
            if len(idx_seg) < 2:  # less than 2 consecutive minutes -> ignore
                fault_event.loc[idx_seg] = False

    fault_event.name = "fault_event"
    df["fault_event"] = fault_event

    # Fault episode = first minute of each contiguous fault run (one incident -> one event)
    prev_fault = fault_event.shift(1, fill_value=False)
    fault_episode = fault_event & ~prev_fault
    fault_episode.name = "fault_episode"
    df["fault_episode"] = fault_episode

    s = df[run_col]
    running = pd.Series(
        np.where(s.isna(), False, s.astype(bool)), index=df.index, dtype=bool
    )
    df[running_mask_col] = running

    # Labels and y_ttf use fault_episode so one incident counts once
    for horizon_min, label_col in [
        (MAINT_HORIZON_1H_MIN, "y_fault_1h"),
        (MAINT_HORIZON_6H_MIN, "y_fault_6h"),
        (MAINT_HORIZON_24H_MIN, "y_fault_24h"),
        (MAINT_HORIZON_3D_MIN, "y_fault_3d"),
    ]:
        label = make_horizon_label(fault_episode, horizon_minutes=horizon_min)
        label = label.where(running, other=False)
        df[label_col] = label.astype(int)

    # Time-to-fault: minutes until next fault *episode* start, capped; only on running minutes
    fault_times = df.index[fault_episode].values.astype("datetime64[ns]")
    idx_ns = df.index.values.astype("datetime64[ns]")
    # For each row t: next fault index = first fault time > t
    # searchsorted(..., side="right") gives first index where fault_times > t
    pos = np.searchsorted(fault_times, idx_ns, side="right")
    minutes_to_fault = np.full(len(df), float(max_ttf_minutes), dtype=np.float64)
    valid = pos < len(fault_times)
    delta = (fault_times[pos[valid]] - idx_ns[valid]).astype("timedelta64[ns]")
    minutes_to_fault[valid] = np.minimum(
        delta.astype(np.float64) / 60.0e9, float(max_ttf_minutes)
    )
    df["y_ttf"] = pd.Series(minutes_to_fault, index=df.index, dtype=float).where(
        running, other=np.nan
    )

    return df


__all__ = [
    "MAINT_HORIZON_1H_MIN",
    "MAINT_HORIZON_6H_MIN",
    "MAINT_HORIZON_24H_MIN",
    "MAINT_HORIZON_3D_MIN",
    "MAINT_HORIZON_7D_MIN",
    "DEFAULT_SHUTDOWN_FAULT_COLUMNS",
    "DEFAULT_FAULT_COLUMNS",
    "detect_run_stop_events",
    "make_horizon_label",
    "add_maintenance_labels",
    "add_fault_horizon_labels",
]
