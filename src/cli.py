"""Simple CLI for fault-in-horizon risk scoring (for derating)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .load_data import build_base_frame
from .label_events import add_fault_horizon_labels
from .predict_demo import RiskThresholds, score_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score gas motor fault-in-horizon risk (6h, 24h, 3d) and time-to-fault for derating."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data"),
        help="Directory containing SCADA CSV files (default: Data)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained models (default: models)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV path (default: predictions.csv)",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only show the latest prediction (for quick checks)",
    )
    parser.add_argument("--coverage", action="store_true", help="Use coverage preset.")
    parser.add_argument("--ultra-coverage", action="store_true", help="Use ultra coverage preset (min_run=1, warmup=0); use same as train.")
    parser.add_argument("--sensitivity", action="store_true", help="Use sensitivity preset (lower thresholds).")
    parser.add_argument("--low", type=float, default=None, help="Risk threshold for low (3d).")
    parser.add_argument("--medium", type=float, default=None, help="Risk threshold for medium (3d).")
    parser.add_argument("--low-6h", type=float, default=None, help="Risk threshold for low, 6h.")
    parser.add_argument("--medium-6h", type=float, default=None, help="Risk threshold for medium, 6h.")
    parser.add_argument("--low-24h", type=float, default=None, help="Risk threshold for low, 24h.")
    parser.add_argument("--medium-24h", type=float, default=None, help="Risk threshold for medium, 24h.")

    args = parser.parse_args()

    model_1h_path = args.models_dir / "model_fault_1h.pkl"
    model_6h_path = args.models_dir / "model_fault_6h.pkl"
    model_24h_path = args.models_dir / "model_fault_24h.pkl"
    model_3d_path = args.models_dir / "model_fault_3d.pkl"
    model_ttf_path = args.models_dir / "model_fault_ttf.pkl"
    model_aggregate_path = args.models_dir / "model_aggregate.pkl"
    feature_names_path = args.models_dir / "feature_names.txt"

    if not all(p.exists() for p in [model_6h_path, model_24h_path, model_3d_path, feature_names_path]):
        print("ERROR: Fault models not found. Run 'python -m src.train_models' first.")
        return

    print(f"Loading data from {args.data_dir}...")
    df = build_base_frame(args.data_dir)
    df = add_fault_horizon_labels(df, run_col="Gen_cb_cld")

    print("Computing predictions...")
    from .features import FeatureConfig

    if args.ultra_coverage:
        config = FeatureConfig.ultra_coverage_preset()
    elif args.coverage:
        config = FeatureConfig.coverage_preset()
    else:
        config = FeatureConfig()
    thresholds = RiskThresholds.sensitivity_preset() if args.sensitivity else RiskThresholds()
    if args.low is not None:
        thresholds.low = args.low
    if args.medium is not None:
        thresholds.medium = args.medium
    if args.low_6h is not None:
        thresholds.low_6h = args.low_6h
    if args.medium_6h is not None:
        thresholds.medium_6h = args.medium_6h
    if args.low_24h is not None:
        thresholds.low_24h = args.low_24h
    if args.medium_24h is not None:
        thresholds.medium_24h = args.medium_24h

    result = score_frame(
        df,
        model_6h_path,
        model_24h_path,
        model_3d_path,
        model_ttf_path if model_ttf_path.exists() else None,
        feature_names_path,
        config=config,
        thresholds=thresholds,
        run_col="Gen_cb_cld",
        label_prefix="fault",
        feature_medians_path=args.models_dir / "feature_medians.json",
        model_1h_path=model_1h_path,
        model_aggregate_path=model_aggregate_path,
    )

    if args.latest_only:
        valid = result.dropna(subset=["p_fault_6h", "p_fault_24h", "p_fault_3d"])
        if len(valid) > 0:
            latest = valid.iloc[-1]
            print(f"\nLatest prediction (at {latest.name}):")
            if "p_fault_1h" in latest and pd.notna(latest.get("p_fault_1h")):
                print(f"  P(fault in 1h):  {latest['p_fault_1h']:.3f} -> {latest['risk_1h']}")
            print(f"  P(fault in 6h):  {latest['p_fault_6h']:.3f} -> {latest['risk_6h']}")
            print(f"  P(fault in 24h): {latest['p_fault_24h']:.3f} -> {latest['risk_24h']}")
            print(f"  P(fault in 3d):  {latest['p_fault_3d']:.3f} -> {latest['risk_3d']}")
            if "minutes_to_fault" in latest and pd.notna(latest.get("minutes_to_fault")):
                print(f"  Minutes to fault: {latest['minutes_to_fault']:.0f} -> {latest['risk_ttf']}")
            if "p_aggregate" in latest and pd.notna(latest.get("p_aggregate")):
                print(f"  P(aggregate):    {latest['p_aggregate']:.3f} -> {latest['risk_aggregate']}")
        else:
            print("No valid predictions found.")
    else:
        result.to_csv(args.output, index_label="Date")
        print(f"\nPredictions saved to {args.output}")
        print(f"Total rows: {len(result)}")
        valid = result.dropna(subset=["p_fault_6h", "p_fault_24h", "p_fault_3d"])
        print(f"Valid predictions: {len(valid)}")
        if len(valid) > 0:
            if "risk_1h" in valid.columns:
                print("\nRisk distribution (1h):")
                print(valid["risk_1h"].value_counts())
            print("\nRisk distribution (6h):")
            print(valid["risk_6h"].value_counts())
            print("\nRisk distribution (24h):")
            print(valid["risk_24h"].value_counts())
            print("\nRisk distribution (3d):")
            print(valid["risk_3d"].value_counts())
            if "risk_ttf" in valid.columns:
                print("\nRisk distribution (ttf):")
                print(valid["risk_ttf"].value_counts())
            if "risk_aggregate" in valid.columns:
                print("\nRisk distribution (aggregate):")
                print(valid["risk_aggregate"].value_counts())


if __name__ == "__main__":
    main()
