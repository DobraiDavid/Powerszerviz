from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
)

from .features import (
    FeatureConfig,
    build_features,
    get_running_and_buffer_blocks,
)
from .label_events import add_fault_horizon_labels, add_maintenance_labels
from .load_data import build_base_frame

UNDERSAMPLE_TARGET_POSITIVE_RATE = 0.25  # for 7d/30d so model can learn
UNDERSAMPLE_RANDOM_STATE = 42


def _time_split(
    index: pd.DatetimeIndex,
    train_fraction: float = 0.8,
    gap_minutes: int = 0,
) -> Tuple[pd.Index, pd.Index]:
    """
    Time-based train/validation split with optional gap to avoid leakage at the boundary.
    If gap_minutes > 0, we exclude that many minutes between train end and val start.
    """
    n = len(index)
    split_at = int(n * train_fraction)
    train_idx = index[:split_at]
    val_start = index[split_at:]
    if gap_minutes > 0 and len(val_start) > 0:
        gap_end = val_start[0] + pd.Timedelta(minutes=gap_minutes)
        val_idx = val_start[val_start >= gap_end]
    else:
        val_idx = val_start
    return train_idx, val_idx


def _undersample_positives(
    y: pd.Series,
    train_idx: pd.Index,
    target_positive_rate: float = UNDERSAMPLE_TARGET_POSITIVE_RATE,
    random_state: int = UNDERSAMPLE_RANDOM_STATE,
) -> pd.Index:
    """
    Return a subset of train_idx so that positive rate is ~target_positive_rate.
    Keeps all negatives; randomly samples positives.
    """
    y_tr = y.loc[train_idx]
    pos_idx = train_idx[y_tr == 1]
    neg_idx = train_idx[y_tr == 0]
    n_neg = len(neg_idx)
    n_pos = len(pos_idx)
    if n_pos == 0:
        return train_idx
    # n_pos_samp / (n_neg + n_pos_samp) = target => n_pos_samp = n_neg * target / (1 - target)
    n_pos_desired = int(round(n_neg * target_positive_rate / (1 - target_positive_rate)))
    n_pos_samp = min(n_pos, max(1, n_pos_desired))
    rng = np.random.default_rng(random_state)
    pos_samp = rng.choice(pos_idx, size=n_pos_samp, replace=False)
    return neg_idx.union(pd.Index(pos_samp))


def _fit_one_model(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    val_idx: pd.Index,
    label_name: str,
    class_weight: Optional[str] = "balanced",
    calibrate: bool = False,
    *,
    max_depth: int = 6,
    max_iter: int = 700,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
    n_iter_no_change: int = 25,
    validation_fraction: float = 0.1,
) -> Tuple[object, Dict]:
    """Fit a classifier; optionally wrap in CalibratedClassifierCV for better probabilities. Returns (model, metrics_dict)."""
    base_clf = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=True,
        n_iter_no_change=n_iter_no_change,
        validation_fraction=validation_fraction,
        class_weight=class_weight,
        random_state=42,
    )
    clf = (
        CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        if calibrate
        else base_clf
    )

    X_tr = X.loc[train_idx]
    y_tr = y.loc[train_idx]
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]

    mask_tr = ~X_tr.isna().any(axis=1)
    X_tr = X_tr[mask_tr]
    y_tr = y_tr[mask_tr]

    mask_val = ~X_val.isna().any(axis=1)
    X_val = X_val[mask_val]
    y_val = y_val[mask_val]

    print(f"{label_name}: Training on {len(X_tr)} samples, validating on {len(X_val)} samples")
    print(f"{label_name}: Positive class rate - train: {y_tr.mean():.4f}, val: {y_val.mean():.4f}")
    if y_val.mean() > 0.95 or y_val.mean() < 0.05:
        print(f"{label_name}: Note: validation set is nearly one class; ROC/PR AUC may be misleading.")

    clf.fit(X_tr, y_tr)

    metrics = {}
    if len(np.unique(y_val)) > 1:
        proba = clf.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, proba)
        pr_auc = average_precision_score(y_val, proba)
        print(f"{label_name}: ROC AUC={roc:.3f}, PR AUC={pr_auc:.3f}")

        prec, rec, thr = precision_recall_curve(y_val, proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.nanargmax(f1)
        best_threshold = float(thr[best_idx])
        best_f1 = float(f1[best_idx])
        best_precision = float(prec[best_idx])
        best_recall = float(rec[best_idx])

        print(
            f"{label_name}: best F1={best_f1:.3f} "
            f"at threshold={best_threshold:.3f} "
            f"(precision={best_precision:.3f}, recall={best_recall:.3f})"
        )

        metrics = {
            "roc_auc": float(roc),
            "pr_auc": float(pr_auc),
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_val)),
            "train_positive_rate": float(y_tr.mean()),
            "val_positive_rate": float(y_val.mean()),
        }
    else:
        print(
            f"{label_name}: Validation set has only one class (no positives in last 20% of timeline); "
            "metrics not computed. Model saved."
        )
        metrics = {
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_val)),
            "train_positive_rate": float(y_tr.mean()),
            "val_positive_rate": float(y_val.mean()),
        }

    return clf, metrics


# Params that _fit_one_model accepts for tuning
_TUNE_PARAMS = ("max_depth", "max_iter", "min_samples_leaf", "l2_regularization", "learning_rate")


def _tune_one_model(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    val_idx: pd.Index,
    label_name: str,
    class_weight: Optional[str] = "balanced",
    n_iter: int = 15,
    calibrate: bool = True,
) -> Dict:
    """Run randomized search; score with calibrated model so we optimize the deployed pipeline."""
    mask_tr = ~X.loc[train_idx].isna().any(axis=1)
    mask_val = ~X.loc[val_idx].isna().any(axis=1)
    X_tr = X.loc[train_idx][mask_tr]
    y_tr = y.loc[X_tr.index]
    X_val = X.loc[val_idx][mask_val]
    y_val = y.loc[X_val.index]
    if len(X_tr) == 0 or len(X_val) == 0 or len(np.unique(y_tr)) < 2:
        return {}

    # Keep min_samples_leaf and l2 modest so we don't over-smooth (avoid 24h recall collapse)
    param_dist = {
        "max_depth": [6, 8, 10],
        "max_iter": [400, 500, 700],
        "min_samples_leaf": [5, 10, 20],
        "learning_rate": [0.05, 0.08],
        "l2_regularization": [0.0, 0.05, 0.1],
    }
    best_score = -1.0
    best_params: Dict = {}
    rng = np.random.default_rng(42)
    keys = list(param_dist.keys())
    for _ in range(n_iter):
        params = {k: rng.choice(param_dist[k]) for k in keys}
        base = HistGradientBoostingClassifier(
            max_depth=params["max_depth"],
            max_iter=params["max_iter"],
            min_samples_leaf=params["min_samples_leaf"],
            learning_rate=params["learning_rate"],
            l2_regularization=params["l2_regularization"],
            class_weight=class_weight,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.1,
        )
        clf = (
            CalibratedClassifierCV(base, method="isotonic", cv=3)
            if calibrate
            else base
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, proba)
        if score > best_score:
            best_score = score
            best_params = {k: params[k] for k in _TUNE_PARAMS if k in params}
    best = {k: v for k, v in best_params.items() if k in _TUNE_PARAMS}
    print(f"{label_name}: Tuning best PR AUC={best_score:.3f} (calibrated), params={best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train fault-in-horizon prediction models (24h, 3d, 7d). Labels = fault/trip signal in next N days (no Oper_off)."
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage preset (use stricter blocks: min_run=20, warmup=10).",
    )
    parser.add_argument(
        "--max-coverage",
        action="store_true",
        help="Use max coverage preset (min_run=3, warmup=2) for more blocks.",
    )
    parser.add_argument(
        "--ultra-coverage",
        action="store_true",
        help="Use ultra coverage preset (min_run=1, warmup=0) to maximize episodes with prediction before.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run randomized hyperparameter search (slower) before fitting each model.",
    )
    parser.add_argument(
        "--maintenance",
        action="store_true",
        help="Train maintenance models (y_maint_1h/6h/24h/3d) instead of fault models.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "Data"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    if args.ultra_coverage:
        # Use ultra coverage only for block building; keep a slightly more stable
        # config for the training mask so we don't train on very short 1-minute runs.
        blocks_config = FeatureConfig.ultra_coverage_preset()
        config = FeatureConfig.max_coverage_preset()
        print(
            "Using ultra coverage blocks (min_run=1, warmup=0) and "
            "max-coverage training config (min_run=3, warmup=2)."
        )
    elif args.max_coverage:
        config = FeatureConfig.max_coverage_preset()
        blocks_config = config
        print("Using max coverage preset (min_run=3, warmup=2).")
    elif args.no_coverage:
        config = FeatureConfig()
        blocks_config = config
        print("Using stricter config (min_run=20, warmup=10).")
    else:
        config = FeatureConfig.coverage_preset()
        blocks_config = config
        print("Using coverage preset (min_run=10, warmup=5) for more prediction rows.")

    print("Loading and merging SCADA data...")
    df = build_base_frame(data_dir)

    is_maint = args.maintenance
    if is_maint:
        print("Adding maintenance labels (1h, 6h, 24h, 3d)...")
        df = add_maintenance_labels(df, run_col="Gen_cb_cld")
        label_prefix = "maint"
    else:
        print("Adding fault-horizon labels (1h, 6h, 24h, 3d + time-to-fault)...")
        df = add_fault_horizon_labels(df, run_col="Gen_cb_cld")
        label_prefix = "fault"

    buffer_minutes = max(config.analog_windows) if config.analog_windows else 60
    blocks = get_running_and_buffer_blocks(
        df,
        run_col="Gen_cb_cld",
        min_run_minutes=blocks_config.min_run_minutes,
        warmup_minutes=blocks_config.warmup_minutes,
        buffer_minutes=buffer_minutes,
    )
    print(f"Building features on {len(blocks)} running+buffer blocks...")

    X_parts = []
    y_1h_parts = []
    y_6h_parts = []
    y_24h_parts = []
    y_3d_parts = []
    mask_parts = []
    for start_ts, end_ts in blocks:
        block_df = df.loc[start_ts:end_ts]
        X_b, y_1h_b, y_6h_b, y_24h_b, y_3d_b, mask_b = build_features(
            block_df, config=config, run_col="Gen_cb_cld", label_prefix=label_prefix
        )
        X_parts.append(X_b)
        y_1h_parts.append(y_1h_b)
        y_6h_parts.append(y_6h_b)
        y_24h_parts.append(y_24h_b)
        y_3d_parts.append(y_3d_b)
        mask_parts.append(mask_b)

    X = pd.concat(X_parts, axis=0)
    mask_trn = pd.concat(mask_parts, axis=0)
    y_1h = pd.concat(y_1h_parts, axis=0).where(mask_trn, other=0)
    y_6h = pd.concat(y_6h_parts, axis=0).where(mask_trn, other=0)
    y_24h = pd.concat(y_24h_parts, axis=0).where(mask_trn, other=0)
    y_3d = pd.concat(y_3d_parts, axis=0).where(mask_trn, other=0)
    y_ttf = df.reindex(X.index)["y_ttf"] if not is_maint else None

    idx = X.index[mask_trn]
    # No gap by default so we keep full 20% for validation and comparable metrics
    train_idx, val_idx = _time_split(idx, train_fraction=0.8, gap_minutes=0)
    print(f"Time-based split: train {len(train_idx)} rows, val {len(val_idx)} rows (last 20% of timeline).")

    # Median imputation from training set so we keep more rows (no drop for NaN)
    feature_medians = X.loc[train_idx].median()
    feature_medians = feature_medians.fillna(0.0)
    X = X.fillna(feature_medians)
    name_suffix = "_maint" if is_maint else ""
    medians_path = models_dir / f"feature_medians{name_suffix}.json"
    with open(medians_path, "w") as f:
        json.dump({k: float(v) for k, v in feature_medians.items()}, f)
    print(f"Median imputation: saved to {medians_path.name} (keeps rows with missing values).")

    models = {}
    metrics_all = {}
    key_prefix = "maint_" if is_maint else "fault_"

    def _fit(horizon: str, y_ser: pd.Series, calibrate: bool, **extra_kw: object) -> None:
        label_name = f"y_{key_prefix.rstrip('_')}_{horizon}"
        if args.tune:
            best = _tune_one_model(X, y_ser, train_idx, val_idx, label_name=label_name, class_weight="balanced")
            extra_kw = {**best, **extra_kw}
        model, metrics = _fit_one_model(
            X, y_ser, train_idx, val_idx, label_name=label_name, class_weight="balanced", calibrate=calibrate, **extra_kw
        )
        models[f"{key_prefix}{horizon}"] = model
        metrics_all[f"model_{key_prefix}{horizon}"] = metrics

    # 1h
    print(f"\nTraining model for y_{key_prefix.rstrip('_')}_1h (full training set, calibrated)...")
    _fit("1h", y_1h, calibrate=True)

    # 6h
    print(f"\nTraining model for y_{key_prefix.rstrip('_')}_6h (full training set, calibrated)...")
    _fit("6h", y_6h, calibrate=True)

    # 24h
    print(f"\nTraining model for y_{key_prefix.rstrip('_')}_24h (full training set, calibrated)...")
    _fit("24h", y_24h, calibrate=True)

    # 3d
    print(f"\nTraining model for y_{key_prefix.rstrip('_')}_3d (full training set, calibrated)...")
    _fit("3d", y_3d, calibrate=True)

    if not is_maint:
        # Time-to-fault regression (minutes until next fault, capped)
        print("\nTraining time-to-fault regressor (y_ttf)...")
        ttf_train = train_idx[y_ttf.loc[train_idx].notna()]
        ttf_val = val_idx[y_ttf.loc[val_idx].notna()]
        if len(ttf_train) > 0 and len(ttf_val) > 0:
            reg = HistGradientBoostingRegressor(
                max_depth=6,
                max_iter=700,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=0.0,
                early_stopping=True,
                n_iter_no_change=25,
                validation_fraction=0.1,
                random_state=42,
            )
            X_tr_ttf = X.loc[ttf_train]
            y_tr_ttf = y_ttf.loc[ttf_train]
            X_val_ttf = X.loc[ttf_val]
            y_val_ttf = y_ttf.loc[ttf_val]
            reg.fit(X_tr_ttf, y_tr_ttf)
            pred_val = reg.predict(X_val_ttf)
            mae = mean_absolute_error(y_val_ttf, pred_val)
            print(f"y_ttf: MAE (val) = {mae:.1f} min")
            models["fault_ttf"] = reg
            metrics_all["model_fault_ttf"] = {
                "mae": float(mae),
                "n_train": int(len(ttf_train)),
                "n_val": int(len(ttf_val)),
            }
        else:
            print("y_ttf: Skipped (insufficient non-NaN targets).")
            models["fault_ttf"] = None

        # Aggregate model: uses 1h, 6h, 24h, 3d probs + normalized ttf to predict fault in 6h
        print("\nTraining aggregate model (meta-model from horizon predictions -> fault in 6h)...")
        max_ttf_min = 7 * 24 * 60
        valid_mask = ~X.isna().any(axis=1)
        X_valid = X[valid_mask].copy()
        X_valid = X_valid.fillna(feature_medians)
        p1h = models["fault_1h"].predict_proba(X_valid)[:, 1]
        p6h = models["fault_6h"].predict_proba(X_valid)[:, 1]
        p24h = models["fault_24h"].predict_proba(X_valid)[:, 1]
        p3d = models["fault_3d"].predict_proba(X_valid)[:, 1]
        ttf_raw = y_ttf.reindex(X_valid.index).fillna(max_ttf_min).clip(upper=max_ttf_min)
        ttf_norm = (ttf_raw / float(max_ttf_min)).to_numpy()
        X_agg = np.column_stack([p1h, p6h, p24h, p3d, ttf_norm])
        y_agg = y_6h.reindex(X_valid.index).fillna(0).astype(int).to_numpy()
        agg_train_mask = X_valid.index.isin(train_idx)
        agg_val_mask = X_valid.index.isin(val_idx)
        train_pos = np.where(agg_train_mask)[0]
        val_pos = np.where(agg_val_mask)[0]
        if len(train_pos) > 0 and len(val_pos) > 0 and len(np.unique(y_agg[train_pos])) > 1:
            X_agg_tr = X_agg[train_pos]
            y_agg_tr = y_agg[train_pos]
            X_agg_val = X_agg[val_pos]
            y_agg_val = y_agg[val_pos]
            agg_clf = LogisticRegression(C=0.5, max_iter=500, random_state=42, class_weight="balanced")
            agg_clf = CalibratedClassifierCV(agg_clf, method="isotonic", cv=3)
            agg_clf.fit(X_agg_tr, y_agg_tr)
            proba_agg_val = agg_clf.predict_proba(X_agg_val)[:, 1]
            roc_agg = roc_auc_score(y_agg_val, proba_agg_val)
            pr_agg = average_precision_score(y_agg_val, proba_agg_val)
            print(f"aggregate: ROC AUC={roc_agg:.3f}, PR AUC={pr_agg:.3f} (val)")
            models["aggregate"] = agg_clf
            metrics_all["model_aggregate"] = {
                "roc_auc": float(roc_agg),
                "pr_auc": float(pr_agg),
                "n_train": int(len(X_agg_tr)),
                "n_val": int(len(X_agg_val)),
            }
        else:
            models["aggregate"] = None
            metrics_all["model_aggregate"] = {}
    elif is_maint:
        # Maintenance aggregate: (p_maint_1h, p_maint_6h, p_maint_24h, p_maint_3d) -> maintenance in 6h
        print("\nTraining maintenance aggregate model (meta-model from horizon predictions -> maint in 6h)...")
        valid_mask = ~X.isna().any(axis=1)
        X_valid = X[valid_mask].copy()
        X_valid = X_valid.fillna(feature_medians)
        p1h = models["maint_1h"].predict_proba(X_valid)[:, 1]
        p6h = models["maint_6h"].predict_proba(X_valid)[:, 1]
        p24h = models["maint_24h"].predict_proba(X_valid)[:, 1]
        p3d = models["maint_3d"].predict_proba(X_valid)[:, 1]
        X_agg = np.column_stack([p1h, p6h, p24h, p3d])
        y_agg = y_6h.reindex(X_valid.index).fillna(0).astype(int).to_numpy()
        agg_train_mask = X_valid.index.isin(train_idx)
        agg_val_mask = X_valid.index.isin(val_idx)
        train_pos = np.where(agg_train_mask)[0]
        val_pos = np.where(agg_val_mask)[0]
        if len(train_pos) > 0 and len(val_pos) > 0 and len(np.unique(y_agg[train_pos])) > 1:
            X_agg_tr = X_agg[train_pos]
            y_agg_tr = y_agg[train_pos]
            X_agg_val = X_agg[val_pos]
            y_agg_val = y_agg[val_pos]
            agg_clf = LogisticRegression(C=0.5, max_iter=500, random_state=42, class_weight="balanced")
            agg_clf = CalibratedClassifierCV(agg_clf, method="isotonic", cv=3)
            agg_clf.fit(X_agg_tr, y_agg_tr)
            proba_agg_val = agg_clf.predict_proba(X_agg_val)[:, 1]
            roc_agg = roc_auc_score(y_agg_val, proba_agg_val)
            pr_agg = average_precision_score(y_agg_val, proba_agg_val)
            print(f"maint aggregate: ROC AUC={roc_agg:.3f}, PR AUC={pr_agg:.3f} (val)")
            models["maint_aggregate"] = agg_clf
            metrics_all["model_maint_aggregate"] = {
                "roc_auc": float(roc_agg),
                "pr_auc": float(pr_agg),
                "n_train": int(len(X_agg_tr)),
                "n_val": int(len(X_agg_val)),
            }
        else:
            models["maint_aggregate"] = None
            metrics_all["model_maint_aggregate"] = {}

    print("\nSaving models...")
    joblib.dump(models[f"{key_prefix}1h"], models_dir / f"model_{key_prefix}1h.pkl")
    joblib.dump(models[f"{key_prefix}6h"], models_dir / f"model_{key_prefix}6h.pkl")
    joblib.dump(models[f"{key_prefix}24h"], models_dir / f"model_{key_prefix}24h.pkl")
    joblib.dump(models[f"{key_prefix}3d"], models_dir / f"model_{key_prefix}3d.pkl")
    if not is_maint:
        if models.get("fault_ttf") is not None:
            joblib.dump(models["fault_ttf"], models_dir / "model_fault_ttf.pkl")
        if models.get("aggregate") is not None:
            joblib.dump(models["aggregate"], models_dir / "model_aggregate.pkl")
    elif is_maint and models.get("maint_aggregate") is not None:
        joblib.dump(models["maint_aggregate"], models_dir / "model_maint_aggregate.pkl")

    feature_names_path = models_dir / f"feature_names{name_suffix}.txt"
    feature_names_path.write_text("\n".join(X.columns))

    horizons_meta = ["1h", "6h", "24h", "3d", "ttf", "aggregate"] if not is_maint else ["1h", "6h", "24h", "3d", "aggregate"]
    eval_metrics = {
        **metrics_all,
        "target": "maintenance" if is_maint else "fault",
        "n_features": int(len(X.columns)),
        "train_split_date": str(train_idx[-1]) if len(train_idx) > 0 else None,
        "val_split_date": str(val_idx[0]) if len(val_idx) > 0 else None,
        "horizons": horizons_meta,
        "coverage_preset": not args.no_coverage,
        "max_coverage_preset": args.max_coverage,
        "ultra_coverage_preset": args.ultra_coverage,
    }
    metrics_path = models_dir / f"evaluation_metrics{name_suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")

    print("\nDone. Models saved to:", models_dir)


if __name__ == "__main__":
    main()
