"""Review predictions against fault or maintenance events to tune thresholds."""
from pathlib import Path

import argparse
import numpy as np
import pandas as pd

from src.load_data import build_base_frame
from src.label_events import add_fault_horizon_labels, add_maintenance_labels

parser = argparse.ArgumentParser(description="Review fault or maintenance predictions vs events.")
parser.add_argument("--maintenance", action="store_true", help="Review maintenance predictions (predictions_maint.csv vs maintenance_event).")
args = parser.parse_args()

is_maint = args.maintenance
pred_path = Path("predictions_maint.csv" if is_maint else "predictions.csv")
if not pred_path.exists():
    print(f"No {pred_path.name} found. Run python -m src.predict_demo" + (" --maintenance" if is_maint else "") + " first.")
    exit(1)

print("Loading predictions...")
preds = pd.read_csv(pred_path, index_col="Date", parse_dates=True)
print(f"Loaded {len(preds)} prediction rows")

if is_maint:
    print("\nLoading SCADA data and maintenance events...")
    df = build_base_frame(Path("Data"))
    df = add_maintenance_labels(df, run_col="Gen_cb_cld")
    cols = ["maintenance_event", "Gen_cb_cld"]
    event_col = "maintenance_event"
    pred_cols_all = ["p_maint_1h", "p_maint_6h", "p_maint_24h", "p_maint_3d", "p_maint_aggregate"]
    pred_cols_valid_list = ["p_maint_6h", "p_maint_24h", "p_maint_3d"]
    event_label = "maintenance events"
else:
    print("\nLoading SCADA data and fault events (Leallas_zav, Gen_trip)...")
    df = build_base_frame(Path("Data"))
    df = add_fault_horizon_labels(df, run_col="Gen_cb_cld")
    cols = ["fault_event", "Gen_cb_cld"]
    event_col = "fault_episode" if "fault_episode" in df.columns else "fault_event"
    pred_cols_all = ["p_fault_1h", "p_fault_6h", "p_fault_24h", "p_fault_3d", "p_aggregate"]
    pred_cols_valid_list = ["p_fault_6h", "p_fault_24h", "p_fault_3d"]
    event_label = "fault episodes" if event_col == "fault_episode" else "fault events"

# Only join columns from df that are not already in preds (predict_demo may have written them to CSV)
cols_from_df = [c for c in cols if c in df.columns and c not in preds.columns]
merged = preds.join(df[cols_from_df], how="inner") if cols_from_df else preds.copy()

pred_cols = [c for c in pred_cols_all if c in merged.columns]
pred_cols_valid = [c for c in pred_cols_valid_list if c in merged.columns]
valid_preds = merged.dropna(subset=pred_cols_valid) if pred_cols_valid else merged
s = merged.loc[valid_preds.index, "Gen_cb_cld"]
running = pd.Series(np.where(s.isna(), False, s.astype(bool)), index=s.index, dtype=bool)
valid_running = valid_preds[running]

print(f"\nValid predictions during running periods: {len(valid_running)}")
print("  (Total prediction rows we scored while the motor was running.)")
if len(valid_running) == 0:
    print("No valid predictions found.")
    exit(1)

event_times = df.index[df[event_col] == True]
total_events = len(event_times)
print(f"\nTotal {event_label}: {total_events}")
if not is_maint:
    print("  (Episode = first minute of each contiguous fault run; one incident counts once.)" if event_col == "fault_episode" else "  (Minutes when Leallas_zav or Gen_trip was True.)")
print("\n  The numbers below count: of these events, how many had at least one prediction")
print("  row in the 6h / 24h / 3d *before* the event start.")

# Column names for prob/risk (fault vs maintenance)
p1h_col = "p_maint_1h" if is_maint else "p_fault_1h"
p6h_col = "p_maint_6h" if is_maint else "p_fault_6h"
p24h_col = "p_maint_24h" if is_maint else "p_fault_24h"
p3d_col = "p_maint_3d" if is_maint else "p_fault_3d"
risk_1h_col = "risk_maint_1h" if is_maint else "risk_1h"
risk_6h_col = "risk_maint_6h" if is_maint else "risk_6h"
risk_24h_col = "risk_maint_24h" if is_maint else "risk_24h"
risk_3d_col = "risk_maint_3d" if is_maint else "risk_3d"
p_agg_col = "p_maint_aggregate" if is_maint else "p_aggregate"
risk_agg_col = "risk_maint_aggregate" if is_maint else "risk_aggregate"
ttf_col = None if is_maint else "minutes_to_fault"

# Diagnostic: upper bound on episodes that can have a prediction in the 6h before (motor must run ≥1 min in that window)
events_with_running_6h = 0
for T in event_times:
    window_6h_end = T - pd.Timedelta(minutes=1)
    window_6h_start = T - pd.Timedelta(hours=6)
    in_window = df.loc[window_6h_start:window_6h_end]
    if "Gen_cb_cld" in in_window.columns:
        s_run = in_window["Gen_cb_cld"]
        # Avoid deprecated downcasting on fillna for object dtypes
        running_in_window = pd.Series(
            np.where(s_run.isna(), False, s_run.astype(bool)),
            index=s_run.index,
            dtype=bool,
        )
        if running_in_window.any():
            events_with_running_6h += 1
print(f"\nEvents with motor running at least 1 minute in the 6h before: {events_with_running_6h}/{total_events}")
print("  (Theoretical maximum: we cannot have 'prediction before' for more than this without scoring during motor off.)")

pred_index = preds.dropna(subset=pred_cols_valid).index if pred_cols_valid else preds.index

print("\n" + "=" * 60)
print("Analysis: Predictions before " + ("maintenance events" if is_maint else "fault episodes"))
print("=" * 60)

events_with_pred_1h = 0
events_preceded_by_high_1h = 0
events_with_pred_6h = 0
events_preceded_by_high_6h = 0
events_with_pred_24h = 0
events_preceded_by_high_24h = 0
events_with_pred_3d = 0
events_preceded_by_high_3d = 0
events_with_pred_agg = 0
events_preceded_by_high_agg = 0
samples = []

for T in event_times:
    if p1h_col in preds.columns:
        window_1h_end = T - pd.Timedelta(minutes=1)
        window_1h_start = T - pd.Timedelta(hours=1)
        in_1h = pred_index[(pred_index >= window_1h_start) & (pred_index <= window_1h_end)]
        if len(in_1h) > 0:
            events_with_pred_1h += 1
            p1 = preds.loc[in_1h]
            if risk_1h_col in preds.columns and (p1[risk_1h_col] == "high").any():
                events_preceded_by_high_1h += 1

    if p6h_col in preds.columns:
        window_6h_end = T - pd.Timedelta(minutes=1)
        window_6h_start = T - pd.Timedelta(hours=6)
        in_6h = pred_index[(pred_index >= window_6h_start) & (pred_index <= window_6h_end)]
        if len(in_6h) > 0:
            events_with_pred_6h += 1
            p6 = preds.loc[in_6h]
            if risk_6h_col in preds.columns and (p6[risk_6h_col] == "high").any():
                events_preceded_by_high_6h += 1
            if len(samples) < 5:
                mean_ttf = p6[ttf_col].mean() if ttf_col and ttf_col in p6.columns else None
                samples.append({
                    "time": T,
                    "mean_p6h": p6[p6h_col].mean() if p6h_col in p6.columns else None,
                    "max_p6h": p6[p6h_col].max() if p6h_col in p6.columns else None,
                    "mean_ttf": mean_ttf,
                    "high_6h": (p6[risk_6h_col] == "high").any() if risk_6h_col in p6.columns else False,
                })

    if p24h_col in preds.columns:
        window_24h_end = T - pd.Timedelta(minutes=1)
        window_24h_start = T - pd.Timedelta(hours=24)
        in_24h = pred_index[(pred_index >= window_24h_start) & (pred_index <= window_24h_end)]
        if len(in_24h) > 0:
            events_with_pred_24h += 1
            p24 = preds.loc[in_24h]
            if risk_24h_col in preds.columns and (p24[risk_24h_col] == "high").any():
                events_preceded_by_high_24h += 1

    if p3d_col in preds.columns:
        window_3d_end = T - pd.Timedelta(minutes=1)
        window_3d_start = T - pd.Timedelta(days=3)
        in_3d = pred_index[(pred_index >= window_3d_start) & (pred_index <= window_3d_end)]
        if len(in_3d) > 0:
            events_with_pred_3d += 1
            p3 = preds.loc[in_3d]
            if risk_3d_col in preds.columns and (p3[risk_3d_col] == "high").any():
                events_preceded_by_high_3d += 1

    if p_agg_col in preds.columns:
        window_6h_end = T - pd.Timedelta(minutes=1)
        window_6h_start = T - pd.Timedelta(hours=6)
        in_agg = pred_index[(pred_index >= window_6h_start) & (pred_index <= window_6h_end)]
        if len(in_agg) > 0:
            events_with_pred_agg += 1
            pag = preds.loc[in_agg]
            if risk_agg_col in preds.columns and (pag[risk_agg_col] == "high").any():
                events_preceded_by_high_agg += 1

if total_events > 0:
    if p1h_col in preds.columns:
        print(f"Events with at least one prediction in the 1h before:   {events_with_pred_1h}/{total_events}")
        if events_with_pred_1h > 0 and risk_1h_col in preds.columns:
            print(f"  Of those, preceded by 'high' risk (1h): {events_preceded_by_high_1h}/{events_with_pred_1h}")
    if p6h_col in preds.columns:
        print(f"Events with at least one prediction in the 6h before:   {events_with_pred_6h}/{total_events}")
        if events_with_pred_6h > 0 and risk_6h_col in preds.columns:
            print(f"  Of those, preceded by 'high' risk (6h): {events_preceded_by_high_6h}/{events_with_pred_6h}")
    if p24h_col in preds.columns:
        print(f"Events with at least one prediction in the 24h before:  {events_with_pred_24h}/{total_events}")
        if events_with_pred_24h > 0 and risk_24h_col in preds.columns:
            print(f"  Of those, preceded by 'high' risk (24h): {events_preceded_by_high_24h}/{events_with_pred_24h}")
    if p3d_col in preds.columns:
        print(f"Events with at least one prediction in the 3 days before: {events_with_pred_3d}/{total_events}")
        if events_with_pred_3d > 0 and risk_3d_col in preds.columns:
            print(f"  Of those, preceded by 'high' risk (3d):  {events_preceded_by_high_3d}/{events_with_pred_3d}")
    if p_agg_col in preds.columns:
        print(f"Events with at least one prediction (aggregate) in the 6h before: {events_with_pred_agg}/{total_events}")
        if events_with_pred_agg > 0 and risk_agg_col in preds.columns:
            print(f"  Of those, preceded by 'high' risk (aggregate): {events_preceded_by_high_agg}/{events_with_pred_agg}")
    if samples:
        win_label = "6h window before event"
        print(f"\nSample ({win_label}):")
        for item in samples:
            p6_str = f"mean P6h={item['mean_p6h']:.3f}, max P6h={item['max_p6h']:.3f}" if item['mean_p6h'] is not None else "N/A"
            ttf_str = f", mean minutes_to_fault={item['mean_ttf']:.0f}" if item.get('mean_ttf') is not None and not np.isnan(item['mean_ttf']) else ""
            print(f"  {item['time']}: {p6_str}{ttf_str}, high_risk={item['high_6h']}")

print("\n" + "=" * 60)
print("Overall prediction statistics")
print("=" * 60)
if p1h_col in valid_running.columns:
    print(f"\nP({'maint' if is_maint else 'fault'} in 1h):")
    print(valid_running[p1h_col].describe())
if p6h_col in valid_running.columns:
    print(f"\nP({'maint' if is_maint else 'fault'} in 6h):")
    print(valid_running[p6h_col].describe())
if p24h_col in valid_running.columns:
    print(f"\nP({'maint' if is_maint else 'fault'} in 24h):")
    print(valid_running[p24h_col].describe())
if p3d_col in valid_running.columns:
    print(f"\nP({'maint' if is_maint else 'fault'} in 3d):")
    print(valid_running[p3d_col].describe())
    if p_agg_col in valid_running.columns:
        print("\nP(aggregate):")
        print(valid_running[p_agg_col].describe())
if ttf_col is not None and ttf_col in valid_running.columns:
    print("\nMinutes to fault (predicted):")
    print(valid_running[ttf_col].describe())
if risk_1h_col in valid_running.columns:
    print("\nRisk distribution (1h):")
    print(valid_running[risk_1h_col].value_counts())
if risk_6h_col in valid_running.columns:
    print("\nRisk distribution (6h):")
    print(valid_running[risk_6h_col].value_counts())
if risk_24h_col in valid_running.columns:
    print("\nRisk distribution (24h):")
    print(valid_running[risk_24h_col].value_counts())
if risk_3d_col in valid_running.columns:
    print("\nRisk distribution (3d):")
    print(valid_running[risk_3d_col].value_counts())
if not is_maint and "risk_ttf" in valid_running.columns:
    print("\nRisk distribution (ttf):")
    print(valid_running["risk_ttf"].value_counts())
if risk_agg_col in valid_running.columns:
    print("\nRisk distribution (aggregate):")
    print(valid_running[risk_agg_col].value_counts())

def _threshold_sweep(
    horizon_name: str,
    proba_col: str,
    window: pd.Timedelta,
    min_coverage: float = 0.90,
) -> None:
    """Print a table of episode-level recall vs minute-level load; recommend threshold for ≥min_coverage with least minutes high."""
    if proba_col not in preds.columns or total_events == 0:
        return
    probs = valid_running[proba_col].dropna()
    if probs.empty:
        return
    total_minutes = len(valid_running)
    quantiles = probs.quantile([0.50, 0.75, 0.90, 0.95, 0.99]).tolist()
    table_thresholds = sorted({round(float(q), 4) for q in quantiles if q and q > 0})
    prob_max = float(probs.max())
    if prob_max <= 0:
        return
    # Finer grid to pick "highest thr with ≥min_coverage" (minimizes minutes_high)
    fine = sorted(set(round(x, 4) for x in np.linspace(0.001, min(prob_max, 0.99), 80)))
    all_thrs = sorted(set(table_thresholds + fine))

    def eval_thr(thr: float) -> tuple[int, int, float, int, float]:
        episodes_high = 0
        covered = 0
        for T in event_times:
            window_end = T - pd.Timedelta(minutes=1)
            window_start = window_end - window
            idx_win = pred_index[(pred_index >= window_start) & (pred_index <= window_end)]
            if len(idx_win) == 0:
                continue
            covered += 1
            p = preds.loc[idx_win, proba_col]
            if (p >= thr).any():
                episodes_high += 1
        minutes_high = int((valid_running[proba_col] >= thr).sum())
        frac_cov = episodes_high / covered if covered else 0.0
        frac_min = minutes_high / total_minutes if total_minutes else 0.0
        return episodes_high, covered, frac_cov, minutes_high, frac_min

    print("\n" + "-" * 60)
    print(f"Threshold sweep for {horizon_name} ({proba_col})")
    print("-" * 60)
    print("thr\tepisodes_high/covered\tfrac_covered\tminutes_high\tfrac_minutes")
    best_thr = None
    best_minutes_high = None
    best_frac_min = None
    for thr in all_thrs:
        episodes_high, covered, frac_cov, minutes_high, frac_min = eval_thr(thr)
        if frac_cov >= min_coverage:
            best_thr = thr
            best_minutes_high = minutes_high
            best_frac_min = frac_min
        if thr in table_thresholds:
            print(f"{thr:.4f}\t{episodes_high}/{covered}\t\t{frac_cov:.3f}\t\t{minutes_high}\t{frac_min:.3f}")

    if best_thr is not None:
        print(f"  → Recommended (≥{min_coverage:.0%} coverage, least minutes high): thr = {best_thr:.4f}  (minutes_high = {best_minutes_high}, frac_min = {best_frac_min:.1%})")
    elif total_events > 0:
        print(f"  → No threshold achieved ≥{min_coverage:.0%} coverage.")

_threshold_sweep("1h", p1h_col, pd.Timedelta(hours=1))
_threshold_sweep("6h", p6h_col, pd.Timedelta(hours=6))
_threshold_sweep("24h", p24h_col, pd.Timedelta(hours=24))
_threshold_sweep("3d", p3d_col, pd.Timedelta(days=3))
if p_agg_col in preds.columns:
    _threshold_sweep("aggregate (6h window)", p_agg_col, pd.Timedelta(hours=6))

print("\n[OK] Threshold review completed")
print("Use the sweep tables above to pick thresholds, then set them in src/predict_demo.py or via CLI.")
