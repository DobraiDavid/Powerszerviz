from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """
    Configuration for feature generation.

    You can adjust window sizes, power threshold, and which columns to treat
    as analog vs binary without changing the main code.
    """

    analog_windows: Sequence[int] = (5, 15, 30, 60)  # minutes; 60 for longer trends
    binary_windows: Sequence[int] = (15, 60)  # minutes
    power_column: str = "Gen_visz_telj"  # generator active power, if available
    # Substrings to pick "key" numeric columns for domain features (rate-of-change, 2h window)
    key_signal_substrings: Sequence[str] = (
        "telj", "hom", "olaj", "hut", "kipuf", "viz", "Gen_tek", "Turbo", "Sziv_lev",
        "Kev_vis", "Mot_olaj", "Gyujt", "nyom", "energ", "fesz", "merhi", "max", "min",
    )
    min_run_minutes: int = 20
    warmup_minutes: int = 10
    power_threshold: Optional[float] = None  # e.g. 0.5 * rated power, if known

    @classmethod
    def coverage_preset(cls) -> "FeatureConfig":
        """
        Preset for good coverage: min_run=10, warmup=5 so more running segments get predictions.
        Use --max-coverage for maximum blocks (min_run=3, warmup=2); use --no-coverage for stricter (min_run=20).
        """
        return cls(min_run_minutes=10, warmup_minutes=5)

    @classmethod
    def max_coverage_preset(cls) -> "FeatureConfig":
        """
        Preset for maximum coverage: very low min_run and warmup so
        many short runs get a block. Use for analysis/coverage targets;
        features on very short segments may be noisier.
        """
        return cls(min_run_minutes=3, warmup_minutes=2)

    @classmethod
    def ultra_coverage_preset(cls) -> "FeatureConfig":
        """
        Preset to maximize how many fault episodes have a prediction in the window before them:
        min_run=1, warmup=0 so every run of 1+ minutes gets a block (and buffer). Use when
        "episodes with prediction before" is the priority; features on 1-min runs are noisier.
        """
        return cls(min_run_minutes=1, warmup_minutes=0)


def _find_bool_and_numeric_columns(
    df: pd.DataFrame,
    *,
    exclude_label_cols: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[str]]:
    bool_cols: List[str] = []
    num_cols: List[str] = []
    default_exclude = {
        "maintenance_event",
        "y_maint_1h", "y_maint_6h", "y_maint_24h", "y_maint_3d", "y_maint_7d", "y_maint_30d",
        "y_fault_1h", "y_fault_6h", "y_fault_24h", "y_fault_3d", "y_fault_7d", "y_fault_30d",
        "y_ttf",
        "fault_event",
        "fault_episode",
        "expected_data",
        "missing_expected",
    }
    exclude = set(exclude_label_cols) if exclude_label_cols else set()
    exclude |= default_exclude

    for col, dtype in df.dtypes.items():
        if col in exclude:
            continue
        if dtype == bool:
            bool_cols.append(col)
        elif np.issubdtype(dtype, np.number):
            num_cols.append(col)

    return bool_cols, num_cols


def mask_running_and_buffer(
    df: pd.DataFrame,
    *,
    run_col: str = "Gen_cb_cld",
    min_run_minutes: int = 20,
    warmup_minutes: int = 10,
    buffer_minutes: int = 60,
) -> pd.Series:
    """
    Boolean mask of rows to keep for feature building: stable running segments
    (after warmup) plus a buffer before each segment so rolling windows have context.

    For each run segment (consecutive run_col == True of length >= min_run_minutes):
      - True from (segment_start + warmup_minutes) to segment end.
      - True from (segment_start - buffer_minutes) to segment start (buffer).
    All other rows False. buffer_minutes should be >= max(analog_windows); 60 is safe.
    """
    if run_col not in df.columns:
        raise ValueError(f"Run-state column '{run_col}' not found.")
    s = df[run_col]
    running = pd.Series(np.where(s.isna(), False, s.astype(bool)), index=df.index, dtype=bool)
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have a DatetimeIndex.")
    mask = pd.Series(False, index=index)
    seg_id = (running != running.shift(1, fill_value=False)).cumsum()

    for s in seg_id[running].unique():
        segment_mask = (seg_id == s) & running
        idx = index[segment_mask]
        if len(idx) < min_run_minutes:
            continue
        segment_start = idx[0]
        segment_end = idx[-1]
        # Stable part: from segment_start + warmup_minutes to segment_end
        keep_stable = idx[warmup_minutes:]
        mask.loc[keep_stable] = True
        # Buffer: up to buffer_minutes before segment_start (so rolling has context)
        try:
            pos = index.get_loc(segment_start)
        except KeyError:
            pos = index.get_indexer([segment_start], method="nearest")[0]
        start_pos = max(0, pos - buffer_minutes)
        buffer_idx = index[start_pos:pos]
        mask.loc[buffer_idx] = True

    return mask


def get_running_and_buffer_blocks(
    df: pd.DataFrame,
    *,
    run_col: str = "Gen_cb_cld",
    min_run_minutes: int = 20,
    warmup_minutes: int = 10,
    buffer_minutes: int = 60,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return contiguous (start, end) timestamp ranges for each "running + buffer" block.
    Used to build features per block so rolling windows do not cross gaps.
    """
    mask = mask_running_and_buffer(
        df, run_col=run_col,
        min_run_minutes=min_run_minutes,
        warmup_minutes=warmup_minutes,
        buffer_minutes=buffer_minutes,
    )
    index = df.index
    kept = index[mask]
    if len(kept) == 0:
        return []
    blocks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    block_start = kept[0]
    for i in range(1, len(kept)):
        delta_min = (kept[i] - kept[i - 1]).total_seconds() / 60.0
        if delta_min > 1.5:  # gap > 1.5 minutes -> new block
            blocks.append((block_start, kept[i - 1]))
            block_start = kept[i]
    blocks.append((block_start, kept[-1]))
    return blocks


def _compute_run_segments(
    running: pd.Series,
    *,
    min_run_minutes: int,
    warmup_minutes: int,
) -> pd.Series:
    """
    Identify timestamps that are in "stable running" periods:

      - consecutive True segments in `running` lasting at least `min_run_minutes`
      - within those, only keep rows starting from `warmup_minutes` after the
        beginning of the segment
    """
    running = pd.Series(np.where(running.isna(), False, running.astype(bool)), index=running.index, dtype=bool)

    # Label consecutive running segments
    seg_id = (running != running.shift(1, fill_value=False)).cumsum()

    # For non-running timestamps, seg_id is still incremented; we only care
    # about segments where running is True.
    mask_stable = pd.Series(False, index=running.index)

    for s in seg_id[running].unique():
        segment_mask = (seg_id == s) & running
        idx = running.index[segment_mask]
        if len(idx) < min_run_minutes:
            continue

        # Discard the first warmup_minutes from this segment
        keep_idx = idx[warmup_minutes:]
        mask_stable.loc[keep_idx] = True

    return mask_stable


def build_training_mask(
    df: pd.DataFrame,
    *,
    config: Optional[FeatureConfig] = None,
    run_col: str = "Gen_cb_cld",
) -> pd.Series:
    """
    Build a boolean mask of rows to use for training:

      - motor is running according to run_col
      - running in a sufficiently long segment (min_run_minutes)
      - past warmup period within each segment
      - (optionally) above a minimum power threshold
      - expected data is present (no communication gaps)
    """
    if config is None:
        config = FeatureConfig()

    if run_col not in df.columns:
        raise ValueError(f"Run-state column '{run_col}' not found.")

    s = df[run_col]
    running = pd.Series(np.where(s.isna(), False, s.astype(bool)), index=df.index, dtype=bool)
    stable_running = _compute_run_segments(
        running,
        min_run_minutes=config.min_run_minutes,
        warmup_minutes=config.warmup_minutes,
    )

    mask = stable_running

    # Filter out minutes where we expected data but have a missing gap
    if "missing_expected" in df.columns:
        miss = df["missing_expected"]
        mask &= ~pd.Series(np.where(miss.isna(), False, miss.astype(bool)), index=df.index, dtype=bool)

    # Optional power threshold filter
    if config.power_threshold is not None and config.power_column in df.columns:
        mask &= df[config.power_column].fillna(0) >= config.power_threshold

    return mask


def build_features(
    df: pd.DataFrame,
    *,
    config: Optional[FeatureConfig] = None,
    run_col: str = "Gen_cb_cld",
    label_prefix: str = "fault",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build rolling-window features and return:

      X    : feature matrix (dataframe)
      y_1h, y_6h, y_24h, y_3d : labels for event in next 1h / 6h / 24h / 3d (prefix from label_prefix)
      mask_trn : boolean mask of rows used for training

    Use label_prefix="fault" (default) when df has y_fault_1h/6h/24h/3d from add_fault_horizon_labels.
    Use label_prefix="maint" when df has y_maint_* from add_maintenance_labels.
    """
    if config is None:
        config = FeatureConfig()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have a DatetimeIndex before feature generation.")

    y1h = f"y_{label_prefix}_1h"
    y6h = f"y_{label_prefix}_6h"
    y24 = f"y_{label_prefix}_24h"
    y3d = f"y_{label_prefix}_3d"
    if y1h not in df.columns or y6h not in df.columns or y24 not in df.columns or y3d not in df.columns:
        raise ValueError(f"Dataframe must contain '{y1h}', '{y6h}', '{y24}' and '{y3d}' labels.")

    bool_cols, num_cols = _find_bool_and_numeric_columns(df)

    exclude_cols = {
        "maintenance_event", "fault_event",
        "y_maint_1h", "y_maint_6h", "y_maint_24h", "y_maint_3d", "y_maint_7d", "y_maint_30d",
        "y_fault_1h", "y_fault_6h", "y_fault_24h", "y_fault_3d", "y_fault_7d", "y_fault_30d",
        "y_ttf",
        "expected_data", "missing_expected",
        run_col,
    }
    bool_cols = [c for c in bool_cols if c not in exclude_cols]
    num_cols = [c for c in num_cols if c not in exclude_cols]

    # Build feature columns in a list, then concat once (avoids fragmented DataFrame)
    parts: List[pd.Series] = []

    # Analog / numeric rolling statistics
    for col in num_cols:
        s = df[col]
        for w in config.analog_windows:
            win = s.rolling(window=w, min_periods=1)
            mean_w = win.mean()
            std_w = win.std()
            parts.append(mean_w.rename(f"{col}_mean_{w}m"))
            parts.append(std_w.rename(f"{col}_std_{w}m"))
            parts.append(win.min().rename(f"{col}_min_{w}m"))
            parts.append(win.max().rename(f"{col}_max_{w}m"))

        step = min(config.analog_windows)
        parts.append((s - s.shift(step)).rename(f"{col}_diff_{step}m"))

        # Deviation from longest-window mean (z-style): helps detect "unusual" vs recent normal
        w_max = max(config.analog_windows)
        mean_max = s.rolling(window=w_max, min_periods=1).mean()
        std_max = s.rolling(window=w_max, min_periods=1).std()
        z = (s - mean_max) / (std_max.fillna(1.0).clip(lower=1e-6) + 1e-6)
        parts.append(z.rename(f"{col}_z_{w_max}m"))

    # Time-of-day (cyclic): helps if fault rate varies by hour
    if isinstance(df.index, pd.DatetimeIndex):
        hour_frac = (df.index.hour.astype(np.float64) + df.index.minute.astype(np.float64) / 60.0) / 24.0 * (2 * np.pi)
        parts.append(pd.Series(np.sin(hour_frac), index=df.index, name="time_hour_sin"))
        parts.append(pd.Series(np.cos(hour_frac), index=df.index, name="time_hour_cos"))

    # Domain-specific: rate of change and 2h window for key signals (power, temps, oil, coolant, etc.)
    key_sub = [s.lower() for s in config.key_signal_substrings]
    key_num_cols = [c for c in num_cols if any(s in c.lower() for s in key_sub)]
    for col in key_num_cols:
        s = df[col]
        for n in [15, 30, 60]:
            diff_n = s - s.shift(n)
            roc = diff_n / float(n)  # per-minute rate of change
            parts.append(roc.rename(f"{col}_roc_{n}m"))
        w120 = 120
        win120 = s.rolling(window=w120, min_periods=1)
        parts.append(win120.mean().rename(f"{col}_mean_{w120}m"))
        parts.append(win120.std().rename(f"{col}_std_{w120}m"))
        parts.append(win120.min().rename(f"{col}_min_{w120}m"))
        parts.append(win120.max().rename(f"{col}_max_{w120}m"))

    # Binary / alarm features: current value and recent counts
    for col in bool_cols:
        s_raw = df[col]
        s = pd.Series(np.where(s_raw.isna(), 0, s_raw.astype(int)), index=df.index, dtype=np.intp)
        parts.append(s.rename(f"{col}_now"))
        for w in config.binary_windows:
            win = s.rolling(window=w, min_periods=1)
            parts.append(win.sum().rename(f"{col}_count_{w}m"))

    # Fault-derived features: use fault_episode (one per incident) when present, else fault_event
    fault_signal = None
    if "fault_episode" in df.columns:
        fault_signal = df["fault_episode"]
    elif "fault_event" in df.columns:
        fault_signal = df["fault_event"]
    if fault_signal is not None:
        # Minutes since last fault/episode (past only)
        fault_times_ns = df.index[fault_signal].astype(np.int64)
        if len(fault_times_ns) > 0:
            idx_ns = df.index.astype(np.int64).values
            pos = np.searchsorted(fault_times_ns, idx_ns, side="right") - 1
            minutes_since = np.where(
                pos >= 0, (idx_ns - fault_times_ns[pos]) / (1e9 * 60), 1e6
            )
            parts.append(
                pd.Series(minutes_since, index=df.index, name="minutes_since_last_fault", dtype=float)
            )
        else:
            parts.append(
                pd.Series(1e6, index=df.index, name="minutes_since_last_fault", dtype=float)
            )
        # Fault/episode counts in past 24h, 3d, 7d (shift(1) to avoid leakage)
        shifted = fault_signal.shift(1)
        arr = shifted.to_numpy(dtype=float)
        out = np.zeros_like(arr, dtype=np.intp)
        ok = ~np.isnan(arr)
        out[ok] = arr[ok].astype(np.intp)
        past_fault = pd.Series(out, index=df.index, dtype=np.intp)
        for name, minutes in [("fault_count_24h", 24 * 60), ("fault_count_3d", 3 * 24 * 60), ("fault_count_7d", 7 * 24 * 60)]:
            fault_count = past_fault.rolling(window=minutes, min_periods=1).sum()
            parts.append(fault_count.rename(name))

    feats = pd.concat(parts, axis=1)

    # Training mask
    mask_trn = build_training_mask(df, config=config, run_col=run_col)

    y_1h = df[y1h].astype(int)
    y_6h = df[y6h].astype(int)
    y_24h = df[y24].astype(int)
    y_3d = df[y3d].astype(int)

    return feats, y_1h, y_6h, y_24h, y_3d, mask_trn


__all__ = [
    "FeatureConfig",
    "mask_running_and_buffer",
    "get_running_and_buffer_blocks",
    "build_training_mask",
    "build_features",
]

