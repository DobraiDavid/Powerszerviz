from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .features import (
    FeatureConfig,
    build_features,
    get_running_and_buffer_blocks,
)
from .label_events import add_fault_horizon_labels, add_maintenance_labels
from .load_data import build_base_frame


@dataclass
class RiskThresholds:
    """Horizon-specific mapping from P(fault) to low/medium/high risk.

    - `low` / `medium` are the global defaults (used for 3d unless overridden).
    - `low_1h`, `low_6h`, `low_24h`, `low_aggregate` override the low threshold for that horizon.
    - `medium_*` override the medium; "high" is > medium.
    """

    # Global default (used for 3d horizon)
    low: float = 0.25
    medium: float = 0.50

    # Per-horizon defaults (can be nudged by CLI)
    # 1h: make "high" very rare / extreme by default (your request)
    low_1h: Optional[float] = 0.10
    medium_1h: Optional[float] = 0.30
    # 6h: target ~80–85% of covered episodes, modest high-minute fraction
    # (sweep: 0.2420 -> 100% episodes, 0.2737 -> 70.6%; choose ~0.26 between them)
    low_6h: Optional[float] = 0.13
    medium_6h: Optional[float] = 0.26
    # 24h: target ~80–85% of covered episodes (sweep: 0.1625 -> 80.4%)
    low_24h: Optional[float] = 0.08
    medium_24h: Optional[float] = 0.1625
    # Aggregate (meta-model for 6h risk): aim for ≥85% episode recall (more faults preceded by high risk)
    # (sweep: 0.3333 -> 100% episodes have high in window, 0.60 -> 70.6%; 0.35 gives ~85%+)
    low_aggregate: Optional[float] = 0.25
    medium_aggregate: Optional[float] = 0.35

    def get_low(self, horizon: str) -> float:
        if horizon == "1h" and self.low_1h is not None:
            return self.low_1h
        if horizon == "6h" and self.low_6h is not None:
            return self.low_6h
        if horizon == "24h" and self.low_24h is not None:
            return self.low_24h
        if horizon == "aggregate" and self.low_aggregate is not None:
            return self.low_aggregate
        return self.low

    def get_medium(self, horizon: str) -> float:
        if horizon == "1h" and self.medium_1h is not None:
            return self.medium_1h
        if horizon == "6h" and self.medium_6h is not None:
            return self.medium_6h
        if horizon == "24h" and self.medium_24h is not None:
            return self.medium_24h
        if horizon == "aggregate" and self.medium_aggregate is not None:
            return self.medium_aggregate
        return self.medium

    @classmethod
    def sensitivity_preset(cls) -> "RiskThresholds":
        """Higher sensitivity: lower thresholds so more minutes are 'high'."""
        return cls(
            low=0.15,
            medium=0.35,
            low_1h=0.05,
            medium_1h=0.20,
            low_6h=0.10,
            medium_6h=0.25,
            low_24h=0.05,
            medium_24h=0.12,
            low_aggregate=0.06,
            medium_aggregate=0.15,
        )

    @classmethod
    def maintenance_preset(cls) -> "RiskThresholds":
        """Maintenance thresholds (hardcoded from review_thresholds --maintenance recommended ≥90% coverage, least minutes high)."""
        return cls(
            low=0.10,
            medium=0.12,
            low_1h=0.22,
            medium_1h=0.40,
            low_6h=0.06,
            medium_6h=0.44,
            low_24h=0.06,
            medium_24h=0.44,
            low_aggregate=0.35,
            medium_aggregate=0.45,
        )


def map_risk(prob: float, thresholds: RiskThresholds, horizon: str = "24h") -> str:
    """Map a probability to low / medium / high using horizon-specific thresholds if set."""
    if np.isnan(prob):
        return "unknown"
    low = thresholds.get_low(horizon)
    medium = thresholds.get_medium(horizon)
    if prob <= low:
        return "low"
    if prob <= medium:
        return "medium"
    return "high"


def thresholds_from_metrics(models_dir: Path) -> RiskThresholds:
    """Return default horizon thresholds.

    We keep this helper so the call site stays simple, but for now we rely on
    the hand-tuned defaults above (derived from your sweep targets).
    """
    return RiskThresholds()


def _risk_from_minutes(minutes: float) -> str:
    """Map predicted minutes-to-fault to risk: high if <= 6h, medium if <= 24h, else low."""
    if np.isnan(minutes):
        return "unknown"
    if minutes <= 6 * 60:
        return "high"
    if minutes <= 24 * 60:
        return "medium"
    return "low"


def score_frame(
    df: pd.DataFrame,
    model_6h_path: Path,
    model_24h_path: Path,
    model_3d_path: Path,
    model_ttf_path: Optional[Path],
    feature_names_path: Path,
    *,
    config: Optional[FeatureConfig] = None,
    thresholds: Optional[RiskThresholds] = None,
    run_col: str = "Gen_cb_cld",
    label_prefix: str = "fault",
    feature_medians_path: Optional[Path] = None,
    model_1h_path: Optional[Path] = None,
    model_aggregate_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Score a dataframe with fault-in-horizon predictions (1h, 6h, 24h, 3d), time-to-fault, and optional aggregate.

    Returns dataframe with columns: p_fault_1h (if model_1h), p_fault_6h, p_fault_24h, p_fault_3d,
    minutes_to_fault, p_aggregate (if model_aggregate), risk_1h, risk_6h, risk_24h, risk_3d, risk_ttf, risk_aggregate.
    """
    if config is None:
        config = FeatureConfig()
    if thresholds is None:
        thresholds = RiskThresholds()

    model_1h = joblib.load(model_1h_path) if model_1h_path and model_1h_path.exists() else None
    model_6h = joblib.load(model_6h_path)
    model_24h = joblib.load(model_24h_path)
    model_3d = joblib.load(model_3d_path)
    model_ttf = joblib.load(model_ttf_path) if model_ttf_path and model_ttf_path.exists() else None
    model_aggregate = joblib.load(model_aggregate_path) if model_aggregate_path and model_aggregate_path.exists() else None
    feature_names = feature_names_path.read_text().splitlines()

    buffer_minutes = max(config.analog_windows) if config.analog_windows else 60
    blocks = get_running_and_buffer_blocks(
        df,
        run_col=run_col,
        min_run_minutes=config.min_run_minutes,
        warmup_minutes=config.warmup_minutes,
        buffer_minutes=buffer_minutes,
    )
    X_parts = []
    for start_ts, end_ts in blocks:
        block_df = df.loc[start_ts:end_ts]
        X_b, _y1h, _y6h, _y24h, _y3d, _mask = build_features(
            block_df, config=config, run_col=run_col, label_prefix=label_prefix
        )
        X_parts.append(X_b)

    if not X_parts:
        out_cols = {
            "p_fault_6h": [], "p_fault_24h": [], "p_fault_3d": [],
            "minutes_to_fault": [],
            "risk_6h": [], "risk_24h": [], "risk_3d": [], "risk_ttf": [],
        }
        if model_1h is not None:
            out_cols["p_fault_1h"] = []
            out_cols["risk_1h"] = []
        if model_aggregate is not None:
            out_cols["p_aggregate"] = []
            out_cols["risk_aggregate"] = []
        return pd.DataFrame(out_cols, index=pd.DatetimeIndex([]))

    X = pd.concat(X_parts, axis=0)
    X = X.reindex(columns=feature_names)

    # Use training medians to fill NaN so we can score more rows
    if feature_medians_path is not None and feature_medians_path.exists():
        with open(feature_medians_path) as f:
            medians: Dict[str, float] = json.load(f)
        fill_cols = [c for c in X.columns if c in medians]
        if fill_cols:
            X = X.fillna({c: medians[c] for c in fill_cols})

    valid_mask = ~X.isna().any(axis=1)
    X_valid = X[valid_mask]

    p1h = np.full(len(X), np.nan)
    p6h = np.full(len(X), np.nan)
    p24h = np.full(len(X), np.nan)
    p3d = np.full(len(X), np.nan)
    minutes_to_fault = np.full(len(X), np.nan)

    if len(X_valid) > 0:
        if model_1h is not None:
            p1h[valid_mask.to_numpy()] = model_1h.predict_proba(X_valid)[:, 1]
        p6h[valid_mask.to_numpy()] = model_6h.predict_proba(X_valid)[:, 1]
        p24h[valid_mask.to_numpy()] = model_24h.predict_proba(X_valid)[:, 1]
        p3d[valid_mask.to_numpy()] = model_3d.predict_proba(X_valid)[:, 1]
        if model_ttf is not None:
            raw = model_ttf.predict(X_valid)
            max_ttf_min = 7 * 24 * 60
            minutes_to_fault[valid_mask.to_numpy()] = np.clip(raw, 0.0, float(max_ttf_min))

    # Aggregate model: (p_1h, p_6h, p_24h, p_3d, ttf_norm) -> p_aggregate
    p_aggregate = np.full(len(X), np.nan)
    if model_aggregate is not None and len(X_valid) > 0:
        max_ttf_min = 7 * 24 * 60
        ttf_vals = minutes_to_fault.copy()
        ttf_vals[np.isnan(ttf_vals)] = max_ttf_min
        ttf_norm = np.clip(ttf_vals / max_ttf_min, 0.0, 1.0)
        p1h_in = p1h.copy()
        p1h_in[np.isnan(p1h_in)] = 0.0
        X_agg = np.column_stack([p1h_in, p6h, p24h, p3d, ttf_norm])
        valid_agg = ~np.isnan(X_agg).any(axis=1)
        if valid_agg.any():
            p_aggregate[valid_agg] = model_aggregate.predict_proba(X_agg[valid_agg])[:, 1]

    out = {
        "p_fault_6h": p6h,
        "p_fault_24h": p24h,
        "p_fault_3d": p3d,
        "minutes_to_fault": minutes_to_fault,
        "risk_6h": [map_risk(p, thresholds, "6h") for p in p6h],
        "risk_24h": [map_risk(p, thresholds, "24h") for p in p24h],
        "risk_3d": [map_risk(p, thresholds, "3d") for p in p3d],
        "risk_ttf": [_risk_from_minutes(m) for m in minutes_to_fault],
    }
    if model_1h is not None:
        out["p_fault_1h"] = p1h
        out["risk_1h"] = [map_risk(p, thresholds, "1h") for p in p1h]
    if model_aggregate is not None:
        out["p_aggregate"] = p_aggregate
        out["risk_aggregate"] = [map_risk(p, thresholds, "aggregate") for p in p_aggregate]

    result = pd.DataFrame(out, index=X.index)
    return result


def score_frame_maint(
    df: pd.DataFrame,
    model_1h_path: Path,
    model_6h_path: Path,
    model_24h_path: Path,
    model_3d_path: Path,
    feature_names_path: Path,
    *,
    config: Optional[FeatureConfig] = None,
    thresholds: Optional[RiskThresholds] = None,
    run_col: str = "Gen_cb_cld",
    feature_medians_path: Optional[Path] = None,
    model_aggregate_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Score a dataframe with maintenance-in-horizon predictions (1h, 6h, 24h, 3d) and optional aggregate.
    Returns dataframe with columns: p_maint_1h, p_maint_6h, p_maint_24h, p_maint_3d,
    risk_maint_*, and p_maint_aggregate / risk_maint_aggregate if model_aggregate_path is set.
    """
    if config is None:
        config = FeatureConfig()
    if thresholds is None:
        thresholds = RiskThresholds()

    model_1h = joblib.load(model_1h_path)
    model_6h = joblib.load(model_6h_path)
    model_24h = joblib.load(model_24h_path)
    model_3d = joblib.load(model_3d_path)
    model_aggregate = joblib.load(model_aggregate_path) if model_aggregate_path and model_aggregate_path.exists() else None
    feature_names = feature_names_path.read_text().splitlines()

    buffer_minutes = max(config.analog_windows) if config.analog_windows else 60
    blocks = get_running_and_buffer_blocks(
        df,
        run_col=run_col,
        min_run_minutes=config.min_run_minutes,
        warmup_minutes=config.warmup_minutes,
        buffer_minutes=buffer_minutes,
    )
    X_parts = []
    for start_ts, end_ts in blocks:
        block_df = df.loc[start_ts:end_ts]
        X_b, _y1h, _y6h, _y24h, _y3d, _mask = build_features(
            block_df, config=config, run_col=run_col, label_prefix="maint"
        )
        X_parts.append(X_b)

    if not X_parts:
        out_empty = {
            "p_maint_1h": [], "p_maint_6h": [], "p_maint_24h": [], "p_maint_3d": [],
            "risk_maint_1h": [], "risk_maint_6h": [], "risk_maint_24h": [], "risk_maint_3d": [],
        }
        if model_aggregate is not None:
            out_empty["p_maint_aggregate"] = []
            out_empty["risk_maint_aggregate"] = []
        return pd.DataFrame(out_empty, index=pd.DatetimeIndex([]))

    X = pd.concat(X_parts, axis=0)
    X = X.reindex(columns=feature_names)

    if feature_medians_path is not None and feature_medians_path.exists():
        with open(feature_medians_path) as f:
            medians: Dict[str, float] = json.load(f)
        fill_cols = [c for c in X.columns if c in medians]
        if fill_cols:
            X = X.fillna({c: medians[c] for c in fill_cols})

    valid_mask = ~X.isna().any(axis=1)
    X_valid = X[valid_mask]

    p1h = np.full(len(X), np.nan)
    p6h = np.full(len(X), np.nan)
    p24h = np.full(len(X), np.nan)
    p3d = np.full(len(X), np.nan)

    if len(X_valid) > 0:
        p1h[valid_mask.to_numpy()] = model_1h.predict_proba(X_valid)[:, 1]
        p6h[valid_mask.to_numpy()] = model_6h.predict_proba(X_valid)[:, 1]
        p24h[valid_mask.to_numpy()] = model_24h.predict_proba(X_valid)[:, 1]
        p3d[valid_mask.to_numpy()] = model_3d.predict_proba(X_valid)[:, 1]

    p_aggregate = np.full(len(X), np.nan)
    if model_aggregate is not None and len(X_valid) > 0:
        X_agg = np.column_stack([
            np.nan_to_num(p1h, nan=0.0),
            np.nan_to_num(p6h, nan=0.0),
            np.nan_to_num(p24h, nan=0.0),
            np.nan_to_num(p3d, nan=0.0),
        ])
        valid_agg = ~np.isnan(X_agg).any(axis=1)
        if valid_agg.any():
            p_aggregate[valid_agg] = model_aggregate.predict_proba(X_agg[valid_agg])[:, 1]

    out = {
        "p_maint_1h": p1h,
        "p_maint_6h": p6h,
        "p_maint_24h": p24h,
        "p_maint_3d": p3d,
        "risk_maint_1h": [map_risk(p, thresholds, "1h") for p in p1h],
        "risk_maint_6h": [map_risk(p, thresholds, "6h") for p in p6h],
        "risk_maint_24h": [map_risk(p, thresholds, "24h") for p in p24h],
        "risk_maint_3d": [map_risk(p, thresholds, "3d") for p in p3d],
    }
    if model_aggregate is not None:
        out["p_maint_aggregate"] = p_aggregate
        out["risk_maint_aggregate"] = [map_risk(p, thresholds, "aggregate") for p in p_aggregate]
    return pd.DataFrame(out, index=X.index)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fault-in-horizon predictions (6h, 24h, 3d) and time-to-fault."
    )
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage preset.")
    parser.add_argument("--max-coverage", action="store_true", help="Use max coverage preset.")
    parser.add_argument("--ultra-coverage", action="store_true", help="Use ultra coverage preset (min_run=1, warmup=0); use same as train.")
    parser.add_argument("--low", type=float, default=None, help="Risk threshold for low (3d only).")
    parser.add_argument("--medium", type=float, default=None, help="Risk threshold for medium (3d only).")
    parser.add_argument("--low-6h", type=float, default=None, help="Risk threshold for low, 6h (default 0.20).")
    parser.add_argument("--medium-6h", type=float, default=None, help="Risk threshold for medium, 6h (default 0.40).")
    parser.add_argument("--low-24h", type=float, default=None, help="Risk threshold for low, 24h (default 0.05).")
    parser.add_argument("--medium-24h", type=float, default=None, help="Risk threshold for medium, 24h (default 0.10).")
    parser.add_argument("--sensitivity", action="store_true", help="Use sensitivity preset (lower thresholds for all).")
    parser.add_argument("--maintenance", action="store_true", help="Predict maintenance-in-horizon (output predictions_maint.csv).")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "Data"
    models_dir = base_dir / "models"

    if args.maintenance:
        model_1h_path = models_dir / "model_maint_1h.pkl"
        model_6h_path = models_dir / "model_maint_6h.pkl"
        model_24h_path = models_dir / "model_maint_24h.pkl"
        model_3d_path = models_dir / "model_maint_3d.pkl"
        model_maint_aggregate_path = models_dir / "model_maint_aggregate.pkl"
        feature_names_path = models_dir / "feature_names_maint.txt"
        feature_medians_path = models_dir / "feature_medians_maint.json"
        if not (model_1h_path.exists() and model_6h_path.exists() and model_24h_path.exists() and model_3d_path.exists() and feature_names_path.exists()):
            raise FileNotFoundError(
                "Maintenance models or feature_names_maint.txt not found. Run python -m src.train_models --maintenance first."
            )
    else:
        model_1h_path = models_dir / "model_fault_1h.pkl"
        model_6h_path = models_dir / "model_fault_6h.pkl"
        model_24h_path = models_dir / "model_fault_24h.pkl"
        model_3d_path = models_dir / "model_fault_3d.pkl"
        model_ttf_path = models_dir / "model_fault_ttf.pkl"
        model_aggregate_path = models_dir / "model_aggregate.pkl"
        feature_names_path = models_dir / "feature_names.txt"
        feature_medians_path = models_dir / "feature_medians.json"
        if not (model_6h_path.exists() and model_24h_path.exists() and model_3d_path.exists() and feature_names_path.exists()):
            raise FileNotFoundError(
                "Fault models or feature_names.txt not found. Run python -m src.train_models first."
            )

    print("Loading and merging SCADA data...")
    df = build_base_frame(data_dir)

    if args.maintenance:
        print("Adding maintenance labels (for feature build)...")
        df = add_maintenance_labels(df, run_col="Gen_cb_cld")
    else:
        print("Adding fault-horizon labels (for feature build)...")
        df = add_fault_horizon_labels(df, run_col="Gen_cb_cld")

    if args.ultra_coverage:
        config = FeatureConfig.ultra_coverage_preset()
    elif args.max_coverage:
        config = FeatureConfig.max_coverage_preset()
    elif args.no_coverage:
        config = FeatureConfig()
    else:
        config = FeatureConfig.coverage_preset()

    if args.maintenance:
        thresholds = RiskThresholds.maintenance_preset()
    elif args.sensitivity:
        thresholds = RiskThresholds.sensitivity_preset()
    else:
        thresholds = thresholds_from_metrics(models_dir)

    # CLI overrides last
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

    if args.maintenance:
        print("Computing maintenance risk scores...")
        result = score_frame_maint(
            df,
            model_1h_path,
            model_6h_path,
            model_24h_path,
            model_3d_path,
            feature_names_path,
            config=config,
            thresholds=thresholds,
            run_col="Gen_cb_cld",
            feature_medians_path=feature_medians_path,
            model_aggregate_path=model_maint_aggregate_path,
        )
        context_cols = [c for c in ["maintenance_event", "Gen_cb_cld"] if c in df.columns]
        out_path = base_dir / "predictions_maint.csv"
        col_msg = "p_maint_1h, p_maint_6h, p_maint_24h, p_maint_3d, p_maint_aggregate; risk_maint_1h, risk_maint_6h, risk_maint_24h, risk_maint_3d, risk_maint_aggregate."
    else:
        print("Computing fault risk scores (for derating)...")
        result = score_frame(
            df,
            model_6h_path,
            model_24h_path,
            model_3d_path,
            model_ttf_path,
            feature_names_path,
            config=config,
            thresholds=thresholds,
            run_col="Gen_cb_cld",
            label_prefix="fault",
            feature_medians_path=feature_medians_path,
            model_1h_path=model_1h_path,
            model_aggregate_path=model_aggregate_path,
        )
        context_cols = [c for c in ["fault_event", "fault_episode", "Gen_cb_cld"] if c in df.columns]
        out_path = base_dir / "predictions.csv"
        col_msg = (
            "Columns: p_fault_1h, p_fault_6h, p_fault_24h, p_fault_3d, p_aggregate; "
            "minutes_to_fault; risk_1h, risk_6h, risk_24h, risk_3d, risk_ttf, risk_aggregate (low/medium/high). Use high risk to derate power."
        )

    if context_cols:
        result = result.join(df[context_cols], how="left")

    result.to_csv(out_path, index_label="Date")

    print(f"Predictions written to: {out_path} ({len(result)} rows)")
    print(col_msg)


__all__ = ["RiskThresholds", "map_risk", "score_frame", "score_frame_maint", "main"]

if __name__ == "__main__":
    main()
