## Gas motor fault-in-horizon prediction (6h, 24h, 3d) + time-to-fault

This project predicts **P(fault in 6h/24h/3d)** and **minutes until next fault** (time-to-fault) from current SCADA for **gradual derating**. Output: **p_fault_6h**, **p_fault_24h**, **p_fault_3d**, **minutes_to_fault**, and risk levels (low/medium/high). Labels = fault/trip signals (Leallas_zav, Gen_trip). Not Oper_off.

Section 6 describes optional maintenance (run→stop) labels; default is fault-based so that we don’t count every brief trip as “maintenance” (see Section 6).

### 1. Project structure

- `Data/` – SCADA CSVs: `felsov02_01.csv` … `felsov02_08.csv`, `felsov02_sg.csv`. See **Section 1b** for content.
- `src/load_data.py` – read CSVs, merge on 1‑minute timeline.
- `src/label_events.py` – **fault_episode** (first minute of each contiguous fault run: one incident → one event) and fault_event; labels y_fault_6h/24h/3d and y_ttf from episodes; optional add_maintenance_labels (deprecated for derating).
- `src/features.py` – rolling-window features from all bool/numeric signals; training mask on running+buffer blocks.
- `src/train_models.py` – train fault-in-horizon classifiers (6h, 24h, 3d) and a time-to-fault regressor. Saves model_fault_6h/24h/3d/ttf.pkl to `models/`.
- `src/predict_demo.py` – score frame, write `predictions.csv` (p_fault_6h/24h/3d, minutes_to_fault, risk_*). Reusable `score_frame()`.
- `src/cli.py` – command-line scoring (e.g. `--latest-only`).
- `review_thresholds.py` – compare predictions to fault events (6h/24h/3d windows).
- `src/data_summary.py` – data diagnostics. Run: `python -m src.data_summary [--max-minutes N]`.
- `requirements.txt` – Python dependencies.

**Faster runs:** Use `build_base_frame(..., max_minutes=43200)` (or start_date/end_date) to limit the timeline when loading data.

### 1b. Data files

Each CSV has `Date` (and often `Packtime`); merging is on `Date` (and `Packtime` when present).

- **felsov02_01.csv** – Run state and main alarms: `Gen_cb_cld`, `Oper_on`, `Oper_off`, `Alarm_act`, `Leallas_zav`, `Gen_trip`, `Gen_warn`, `Pls_op_hrs`, `Pls_strt_cnt`, oil/coolant/gas alarms, etc.
- **felsov02_02–06** – Further alarms and analogs (cylinder, cooling, generator, power, exhaust temps, etc.).
- **felsov02_07–08** – Gas/energy and comm status.
- **felsov02_sg.csv** – Communication helper (`Comm_hi`).

The pipeline uses all bool and numeric columns from the merged frame as features (excluding labels and run state). These cover the signal groups relevant for maintenance: alarms/faults, engine/oil/coolant, wear/cycle counts, cylinder/combustion, gas/fuel, generator/electrical, and trending/pump signals.

### 2. Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Training

From the project root:

```bash
python -m src.train_models                # coverage preset (min_run=10, warmup=5)
python -m src.train_models --no-coverage  # stricter blocks (min_run=20, warmup=10)
python -m src.train_models --max-coverage # max coverage (min_run=3, warmup=2)
python -m src.train_models --ultra-coverage # ultra coverage (min_run=1, warmup=0); see Section 8
python -m src.train_models --tune         # optional: hyperparameter search (slower)
```

Training uses a **time-based 80/20 split**: first 80% for training, last 20% for validation. If the last 20% has no positive labels, validation metrics are skipped. 
This loads SCADA from `Data/`, adds maintenance labels (run→stop, not Oper_off, motor off ≥5 min; optional fault columns can be passed to restrict), builds features on running+buffer blocks, time-splits train/val, and trains three models:

- **6h, 24h, 3d:** fault-in-horizon classifiers (full training set, balanced class weight, calibrated).
- **Time-to-fault (y_ttf):** regressor predicting minutes until next fault (capped at 7 days).

Saves:

- `models/model_fault_6h.pkl`, `model_fault_24h.pkl`, `model_fault_3d.pkl`, `model_fault_ttf.pkl` (ttf optional)
- `models/feature_names.txt`
- `models/feature_medians.json` (for imputation at predict time)
- `models/evaluation_metrics.json`

### 4. Predictions

**Full script:**

```bash
python -m src.predict_demo                # coverage preset
python -m src.predict_demo --max-coverage
python -m src.predict_demo --ultra-coverage  # same as train; use when train used --ultra-coverage
python -m src.predict_demo --sensitivity  # lower risk thresholds (more high-risk minutes)
python -m src.predict_demo --low 0.2 --medium 0.5
```

**CLI (quick check):**

```bash
python -m src.cli --latest-only           # print latest P(fault 6h/24h/3d), minutes_to_fault, risk
python -m src.cli --output my_preds.csv
python -m src.cli --ultra-coverage        # use ultra coverage (same as train/predict --ultra-coverage)
```

Output columns:

- `p_fault_6h`, `p_fault_24h`, `p_fault_3d` – P(fault in next 6h / 24h / 3d)
- `minutes_to_fault` – predicted minutes until next fault (from time-to-fault model)
- `risk_6h`, `risk_24h`, `risk_3d`, `risk_ttf` – low / medium / high from configurable thresholds

**Derating:** Use 6h, 24h, and 3d risk (and optionally risk_ttf: high if predicted minutes &lt; 6h). Default thresholds: low=0.25, medium=0.5 (tune with `review_thresholds.py`).

Predictions are produced only for minutes inside “running + buffer” blocks (same as training).

### 5. Review and tune thresholds

```bash
python review_thresholds.py
```

This loads `predictions.csv` and SCADA, identifies fault events, and reports:

- How many events had at least one prediction in the 6h / 24h / 3d before.
- Of those, how many were preceded by “high” risk.
- Overall stats for p_fault_* and risk distributions.

Use this to adjust `RiskThresholds` in `src/predict_demo.py` or use `--low`, `--medium`, `--sensitivity` when running prediction.

### 5b. Understanding the output

**Why “Valid predictions during running periods” (e.g. 6700) is less than total rows (e.g. 12 035) or raw data (~500k)?**

- Raw SCADA has one row per minute (hundreds of thousands of rows). We do **not** predict every minute: we only build features and predict for **“running + buffer” blocks** (segments where the motor runs at least `min_run` minutes, plus a short buffer for rolling windows). So the prediction file has one row per minute **inside those blocks** (e.g. 12 035 rows).
- **“Valid predictions during running periods”** = of those prediction rows, only the ones where at that timestamp the motor was actually **running** (`Gen_cb_cld` True). The rest are in the **buffer** (e.g. warmup) part of the block, where we have a score but the motor is not considered “running” in the review. So 6 700 = number of prediction rows that fall on running minutes; the difference (e.g. 12 035 − 6 700) is buffer/warmup or non-running minutes inside the blocks.

**What “Events with at least one prediction in the 24h before: 402/1423” and “preceded by high risk: 188/402” means**

- **1423** = total fault events (minutes when Leallas_zav or Gen_trip was True).
- **402** = of those 1423 faults, only 402 had **at least one** prediction row in the **24 hours before** the fault. The other 1021 faults had no prediction in that window (motor not running then, or no block covering that time).
- **188/402** = of the 402 faults we *could* have seen (we had predictions in the 24h before), we actually flagged **188** as **“high” risk** in the 24h before the fault. So we “caught” 188 and **missed** 214 (53%) that we had data for. Improving the model or lowering the high-risk threshold (e.g. `--sensitivity`) would increase how many of those 402 are preceded by high risk, at the cost of more false alarms.

**Training metrics (ROC AUC, PR AUC, precision, recall, F1)**

- **ROC AUC** (0–1): Ability to rank “fault soon” higher than “no fault”. Random = 0.5; 0.858 is good separation.
- **PR AUC** (precision–recall AUC): Same idea but better for **imbalanced** data (few faults). Higher is better; 0.574 is moderate.
- **Precision**: Of all rows we predict as positive (e.g. “high risk”), what fraction were actually positive. 0.354 → when we say “high risk”, we’re right ~35% of the time; the rest are false alarms.
- **Recall**: Of all actual positives (faults), what fraction we predicted positive. 0.967 → we catch ~97% of faults but with many false positives.
- **F1**: Harmonic mean of precision and recall. The reported “best F1 at threshold=0.074” is the operating point that balances precision and recall for that validation set.

For derating, you trade off: **higher recall** = catch more faults, **higher precision** = fewer false alarms. Use `review_thresholds.py` and `--low` / `--medium` to tune.

### 6. How maintenance events are defined 
A **maintenance event** is a run→stop that meets:

1. **Not voluntary:** `Oper_off` is False at the stop minute (fault/trip, not operator off).
2. **Fault at stop:** At least one of `shutdown_fault_columns` is True at the stop minute (default: **Leallas_zav**, **Gen_trip**).
3. **Minimum downtime:** The motor stays off for at least **20 minutes** after the stop (until the next run). Brief glitches are excluded.

So by default: **maintenance_event** = run→stop, not Oper_off, one of Leallas_zav/Gen_trip at stop, downtime ≥ 20 min. Pass `shutdown_fault_columns=[]` to not require a fault signal; pass `min_downtime_minutes` to change the downtime threshold.

**Fault labels:** There is no complete mapping of error codes (e.g. 1021, 1023, 1040, 452, 453) to SCADA columns. The pipeline uses **Leallas_zav** and **Gen_trip** by default. To use different columns, pass `fault_columns=...` to `add_fault_horizon_labels()`.

### 7. Reusable scoring

```python
from pathlib import Path
from src.load_data import build_base_frame
from src.label_events import add_fault_horizon_labels
from src.predict_demo import score_frame

df = build_base_frame(Path("Data"))
df = add_fault_horizon_labels(df, run_col="Gen_cb_cld")

result = score_frame(
    df,
    Path("models/model_fault_6h.pkl"),
    Path("models/model_fault_24h.pkl"),
    Path("models/model_fault_3d.pkl"),
    Path("models/model_fault_ttf.pkl"),  # optional; use None if not trained
    Path("models/feature_names.txt"),
)
# result has p_fault_6h, p_fault_24h, p_fault_3d, minutes_to_fault, risk_6h, risk_24h, risk_3d, risk_ttf
```

### 8. Improving the model

**Built-in improvements (already in the pipeline):**

- **Features:** Rolling stats (mean/std/min/max) over 5, 15, 30, 60 min; per-analog **z-score vs 60m**; **time-of-day** (hour sin/cos); **fault-derived** (from fault_episode): `minutes_since_last_fault`, `fault_count_24h/3d/7d`; **domain**: for key signals (power, temps, oil, coolant, etc.) **rate of change** (roc_15m, roc_30m, roc_60m) and **2h window** (mean/std/min/max).
- **Training:** **Early stopping** and light **L2**; **min_samples_leaf=10** so we don’t over-smooth (keeps 24h recall); **CalibratedClassifierCV** for 6h, 24h, and 3d; time-to-fault regressor (y_ttf). No train/val gap by default so the full last 20% is used for validation.
- **Imputation:** Training **median imputation** is saved to `models/feature_medians.json` and used at predict time so more rows get a score (fewer dropped for NaN).
- **Tuning:** **`--tune`** runs a randomized search that scores the **calibrated** model (so best params match what you deploy). Search space is limited (e.g. min_samples_leaf ≤ 20) to avoid overly conservative 24h models.

**What you can still tune:**

- **Thresholds:** Run `review_thresholds.py` and adjust `RiskThresholds` in `src/predict_demo.py`. **6h uses lower defaults** (low_6h=0.12, medium_6h=0.28) so more short-term minutes are flagged; override with `--low-6h`, `--medium-6h`. For 24h/3d use `--low`, `--medium` or the sensitivity preset (`--sensitivity`).
- **Coverage:** Default preset uses `min_run=10`, `warmup=5`. Use `--max-coverage` for more (shorter) blocks; use `--no-coverage` for stricter blocks (min_run=20, warmup=10). Use **`--ultra-coverage`** (min_run=1, warmup=0) when maximizing **“episodes with a prediction in the window before”** is the priority: every run of 1+ minutes gets a block, so more fault episodes will have at least one prediction in the 6h/24h/3d before. **Trade-off:** features on 1‑minute runs are noisier (rolling windows use at most 1 running minute; buffer still gives 60 min of context). Use the same preset for train and predict (e.g. train and predict both with `--ultra-coverage`).

**Fundamental improvements when 24h/3d “high risk” still misses too many faults:**

- **Combine horizons:** Treat “high risk” as **high if (risk_6h == high) OR (risk_24h == high) OR (risk_ttf == high)** so you catch more events (at the cost of more false alarms). Implement this in your derating logic when consuming `predictions.csv`.
- **Fault episodes (done):** Right now every *minute* with Leallas_zav/Gen_trip is a “fault event”, so one incident can count as hundreds of events. For 24h/3d you can instead label **one target per run segment** (e.g. “fault in next 24h” = 1 if this run has any fault in the next 24h). That reduces label noise and can improve the model; it would require a new label function and retrain.
- **Domain features (done):** Key signals get rate-of-change (roc_15m/30m/60m) and 2h window; see `FeatureConfig.key_signal_substrings`.
- **Different horizons:** Try training an extra model for **12h** as a middle ground between 6h and 24h, or use **risk_ttf** (minutes to fault) as the main short-term signal instead of P(fault in 6h).

### 9. Next steps

- Retrain periodically as new SCADA data arrives.
- Use `review_thresholds.py` to tune risk thresholds before using predictions in operations.
