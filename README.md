## Gas motor fault and maintenance prediction

This project predicts **P(fault in 1h/6h/24h/3d)** and **minutes until next fault** (time-to-fault) from SCADA for **gradual derating**, plus an **aggregate** meta-model. A parallel path predicts **unplanned maintenance** (run→stop, fault at stop, downtime ≥ 20 min). Outputs: **p_fault_1h/6h/24h/3d**, **p_aggregate**, **minutes_to_fault**, **risk_*** for fault; **p_maint_1h/6h/24h/3d**, **p_maint_aggregate**, **risk_maint_*** for maintenance. Fault labels = Leallas_zav, Gen_trip; maintenance definition in **Section 6**.

### 1. Project structure

- `Data/` – SCADA CSVs: `felsov02_01.csv` … `felsov02_08.csv`, `felsov02_sg.csv`. See **Section 1b** for content.
- `src/load_data.py` – read CSVs, merge on 1‑minute timeline.
- `src/label_events.py` – **fault_episode** (first minute of each contiguous fault run), **fault_event**; labels **y_fault_1h/6h/24h/3d**, **y_ttf**; **add_maintenance_labels** → **maintenance_event**, **y_maint_1h/6h/24h/3d**.
- `src/features.py` – rolling-window features; training mask on running+buffer blocks; supports `label_prefix="fault"` or `"maint"`.
- `src/train_models.py` – train fault models (1h, 6h, 24h, 3d, ttf, aggregate) or **`--maintenance`** (maint 1h/6h/24h/3d + maint aggregate). Saves to `models/` (e.g. `model_fault_*.pkl`, `model_maint_*.pkl`, `feature_names.txt` / `feature_names_maint.txt`, `feature_medians.json` / `feature_medians_maint.json`, `evaluation_metrics.json` / `evaluation_metrics_maint.json`).
- `src/predict_demo.py` – **fault**: `score_frame()`, writes `predictions.csv`. **`--maintenance`**: `score_frame_maint()`, writes `predictions_maint.csv`. Configurable risk thresholds (defaults, `--sensitivity`, maintenance preset).
- `src/cli.py` – command-line scoring (e.g. `--latest-only`).
- `review_thresholds.py` – compare predictions to fault episodes or **`--maintenance`** to maintenance events; threshold sweeps and **recommended (≥90% coverage, least minutes high)** per horizon.
- `src/view_predictions_app.py` – Streamlit app: **Fault** or **Maintenance** mode, day view, risk and probabilities per minute.
- `src/data_summary.py` – data diagnostics. Run: `python -m src.data_summary [--max-minutes N]`.
- `scripts/build_report_docx.py` – generate Hungarian Word report (`Elorejelzes_jelentes.docx`) from current results.
- `requirements.txt` – Python dependencies (includes `python-docx` for report).

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
python -m src.train_models --maintenance  # train maintenance models (y_maint_1h/6h/24h/3d + aggregate)
python -m src.train_models --ultra-coverage --maintenance  # maintenance with max coverage
```

Training uses a **time-based 80/20 split**: first 80% for training, last 20% for validation. If the last 20% has no positive labels, validation metrics are skipped.

**Fault path (default):** Loads SCADA from `Data/`, adds **fault** labels (Leallas_zav, Gen_trip → fault_episode, y_fault_1h/6h/24h/3d, y_ttf), builds features on running+buffer blocks, and trains:

- **1h, 6h, 24h, 3d:** fault-in-horizon classifiers (calibrated).
- **Time-to-fault (y_ttf):** regressor predicting minutes until next fault (capped at 7 days).
- **Aggregate:** meta-model from (p_1h, p_6h, p_24h, p_3d, ttf_norm) → P(fault in 6h).

**Maintenance path (`--maintenance`):** Uses **add_maintenance_labels** (run→stop, not Oper_off, fault at stop, downtime ≥ 20 min), same feature build with `label_prefix="maint"`, and trains 1h/6h/24h/3d classifiers plus **maint aggregate** (no TTF).

Saves:

- **Fault:** `models/model_fault_1h.pkl`, `model_fault_6h.pkl`, `model_fault_24h.pkl`, `model_fault_3d.pkl`, `model_fault_ttf.pkl`, `model_aggregate.pkl`, `feature_names.txt`, `feature_medians.json`, `evaluation_metrics.json`
- **Maintenance:** `models/model_maint_1h.pkl`, … `model_maint_3d.pkl`, `model_maint_aggregate.pkl`, `feature_names_maint.txt`, `feature_medians_maint.json`, `evaluation_metrics_maint.json`

### 4. Predictions

**Fault (default):**

```bash
python -m src.predict_demo                # coverage preset
python -m src.predict_demo --max-coverage
python -m src.predict_demo --ultra-coverage  # same as train; use when train used --ultra-coverage
python -m src.predict_demo --sensitivity  # lower risk thresholds (more high-risk minutes)
python -m src.predict_demo --low 0.2 --medium 0.5
python -m src.predict_demo --low-6h 0.13 --medium-6h 0.26 --medium-aggregate 0.35  # per-horizon overrides
```

**Maintenance:**

```bash
python -m src.predict_demo --maintenance
python -m src.predict_demo --maintenance --ultra-coverage
```

**CLI (quick check, fault only):**

```bash
python -m src.cli --latest-only           # print latest P(fault 6h/24h/3d), minutes_to_fault, risk
python -m src.cli --output my_preds.csv
python -m src.cli --ultra-coverage
```

**Output:**

- **predictions.csv (fault):** `p_fault_1h`, `p_fault_6h`, `p_fault_24h`, `p_fault_3d`, `p_aggregate`, `minutes_to_fault`, `risk_1h`, `risk_6h`, `risk_24h`, `risk_3d`, `risk_ttf`, `risk_aggregate`, plus `fault_event`, `fault_episode`, `Gen_cb_cld` when present.
- **predictions_maint.csv (maintenance):** `p_maint_1h`, `p_maint_6h`, `p_maint_24h`, `p_maint_3d`, `p_maint_aggregate`, `risk_maint_1h`, … `risk_maint_aggregate`, plus `maintenance_event`, `Gen_cb_cld`.

**Derating:** Use 6h, 24h, 3d and/or **risk_aggregate** (fault) or **risk_maint_aggregate** (maintenance). Fault aggregate defaults: low ≤ 0.34, high &gt; 0.35 (tune with `review_thresholds.py`). Maintenance uses a dedicated preset (≥90% event coverage, least minutes high); override via `--low-*` / `--medium-*` if needed.

Predictions are produced only for minutes inside “running + buffer” blocks (same as training).

### 5. Review and tune thresholds

```bash
python review_thresholds.py                # fault: predictions.csv vs fault episodes
python review_thresholds.py --maintenance  # maintenance: predictions_maint.csv vs maintenance_event
```

Each run loads the corresponding predictions file and SCADA, identifies **fault episodes** (or **maintenance events**), and reports:

- How many events had at least one prediction in the 1h / 6h / 24h / 3d before (and aggregate for 6h window).
- Of those, how many were preceded by “high” risk.
- Overall stats for p_fault_* / p_maint_* and risk distributions.
- **Threshold sweep** per horizon: for several probability cutoffs, episodes covered and fraction of minutes labeled “high”.
- **Recommended (≥90% coverage, least minutes high):** the highest threshold that still achieves ≥90% of events with at least one prediction above that threshold (minimizes false-alarm minutes).

Use the sweep and recommended line to set `RiskThresholds` in `src/predict_demo.py` or override with `--low-*`, `--medium-*`, `--sensitivity` when running prediction.

**Viewer (Streamlit):**

```bash
streamlit run src/view_predictions_app.py
```

Sidebar: choose **Fault** or **Maintenance**. Fault mode shows `predictions.csv` (p_fault_*, risk_*, minutes_to_fault, fault_event, fault_episode). Maintenance mode shows `predictions_maint.csv` (p_maint_*, risk_maint_*, maintenance_event). You can step by minute and see a day overview with events highlighted.

### 5b. Understanding the output

**Why “Valid predictions during running periods” (e.g. 10 780) is less than total prediction rows (e.g. 96 130) or raw data (~500k)?**

- Raw SCADA has one row per minute (hundreds of thousands of rows). We do **not** predict every minute: we only build features and predict for **“running + buffer” blocks** (segments where the motor runs at least `min_run` minutes, plus a buffer before each segment). So the prediction file has one row per minute **inside those blocks** (e.g. 96 130 rows).
- **“Valid predictions during running periods”** = of those prediction rows, only the ones where at that timestamp the motor was actually **running** (`Gen_cb_cld` True). The rest are in the **buffer** (e.g. up to 60 min before a run), where we have a score but the motor is not considered “running” in the review. So e.g. 10 780 = number of prediction rows that fall on running minutes; the difference is buffer or non-running minutes inside the blocks.

**What “Total fault episodes: 121” and “Events with at least one prediction in the 24h before: 51/121”, “preceded by high risk (24h): 41/51” means**

- **121** = total **fault episodes** (first minute of each contiguous fault run; one incident = one episode).
- **51** = of those 121 episodes, 51 had **at least one** prediction row in the **24 hours before** the episode. The others had no prediction in that window (motor not running then, or no block covering that time).
- **41/51** = of the 51 episodes we *could* have seen, we actually had **“high” risk (24h)** in the 24h before for 41. Lowering the high-risk threshold (e.g. `--sensitivity`) or using the recommended threshold from the sweep increases how many are preceded by high risk, at the cost of more minutes labeled high (more false alarms).

**Training metrics (ROC AUC, PR AUC, precision, recall, F1)**

- **ROC AUC** (0–1): Ability to rank “fault soon” higher than “no fault”. Random = 0.5; 0.858 is good separation.
- **PR AUC** (precision–recall AUC): Same idea but better for **imbalanced** data (few faults). Higher is better; 0.574 is moderate.
- **Precision**: Of all rows we predict as positive (e.g. “high risk”), what fraction were actually positive. 0.354 → when we say “high risk”, we’re right ~35% of the time; the rest are false alarms.
- **Recall**: Of all actual positives (faults), what fraction we predicted positive. 0.967 → we catch ~97% of faults but with many false positives.
- **F1**: Harmonic mean of precision and recall. The reported “best F1 at threshold=0.074” is the operating point that balances precision and recall for that validation set.

For derating, you trade off: **higher recall** = catch more faults, **higher precision** = fewer false alarms. Use `review_thresholds.py` and `--low` / `--medium` to tune.

### 6. How fault and maintenance are defined

**Fault:** A **fault event** is any minute where at least one of **Leallas_zav**, **Gen_trip** is True. Contiguous fault runs shorter than 2 minutes are dropped. A **fault episode** is the **first minute** of each contiguous fault run (one incident → one episode). Labels **y_fault_1h/6h/24h/3d** = 1 if any fault episode starts in that future window; **y_ttf** = minutes until next episode (capped at 7 days). **Gen_cb_cld** = motor running; predictions are only for running (and buffer) minutes.

**Maintenance:** A **maintenance event** is a run→stop that meets:

1. **Not voluntary:** `Oper_off` is False at the stop minute (fault/trip, not operator off).
2. **Fault at stop:** At least one of `shutdown_fault_columns` is True at the stop minute (default: **Leallas_zav**, **Gen_trip**).
3. **Minimum downtime:** The motor stays off for at least **20 minutes** after the stop. Brief glitches are excluded.

So by default: **maintenance_event** = run→stop, not Oper_off, one of Leallas_zav/Gen_trip at stop, downtime ≥ 20 min. The **maintenance path** (train `--maintenance`, predict `--maintenance`, review `--maintenance`) uses this definition and trains/evals **y_maint_1h/6h/24h/3d** and an aggregate model.

**Fault columns:** There is no complete mapping of error codes to SCADA columns. The pipeline uses **Leallas_zav** and **Gen_trip** by default. Pass `fault_columns=...` to `add_fault_horizon_labels()` or `shutdown_fault_columns=[]` in `add_maintenance_labels()` to relax the fault-at-stop requirement.

### 7. Reusable scoring

**Fault:**

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
    Path("models/model_fault_ttf.pkl"),
    Path("models/feature_names.txt"),
    model_1h_path=Path("models/model_fault_1h.pkl"),
    model_aggregate_path=Path("models/model_aggregate.pkl"),
    feature_medians_path=Path("models/feature_medians.json"),
)
# result: p_fault_1h/6h/24h/3d, p_aggregate, minutes_to_fault, risk_1h/6h/24h/3d/ttf/aggregate
```

**Maintenance:**

```python
from src.label_events import add_maintenance_labels
from src.predict_demo import score_frame_maint

df = add_maintenance_labels(build_base_frame(Path("Data")), run_col="Gen_cb_cld")
result = score_frame_maint(
    df,
    Path("models/model_maint_1h.pkl"),
    Path("models/model_maint_6h.pkl"),
    Path("models/model_maint_24h.pkl"),
    Path("models/model_maint_3d.pkl"),
    Path("models/feature_names_maint.txt"),
    feature_medians_path=Path("models/feature_medians_maint.json"),
    model_aggregate_path=Path("models/model_maint_aggregate.pkl"),
)
# result: p_maint_1h/6h/24h/3d, p_maint_aggregate, risk_maint_*
```

### 8. Improving the model

**Built-in improvements (already in the pipeline):**

- **Features:** Rolling stats (mean/std/min/max) over 5, 15, 30, 60 min; per-analog **z-score vs 60m**; **time-of-day** (hour sin/cos); **fault-derived** (from fault_episode): `minutes_since_last_fault`, `fault_count_24h/3d/7d`; **domain**: for key signals (power, temps, oil, coolant, etc.) **rate of change** (roc_15m, roc_30m, roc_60m) and **2h window** (mean/std/min/max).
- **Training:** **Early stopping** and light **L2**; **min_samples_leaf=10** so we don’t over-smooth (keeps 24h recall); **CalibratedClassifierCV** for 6h, 24h, and 3d; time-to-fault regressor (y_ttf). No train/val gap by default so the full last 20% is used for validation.
- **Imputation:** Training **median imputation** is saved to `models/feature_medians.json` and used at predict time so more rows get a score (fewer dropped for NaN).
- **Tuning:** **`--tune`** runs a randomized search that scores the **calibrated** model (so best params match what you deploy). Search space is limited (e.g. min_samples_leaf ≤ 20) to avoid overly conservative 24h models.

**What you can still tune:**

- **Thresholds:** Run `review_thresholds.py` (or `--maintenance` for maintenance). Adjust `RiskThresholds` in `src/predict_demo.py` or use `--low-*`, `--medium-*`, `--sensitivity`. Fault **aggregate** defaults: low ≤ 0.34, high > 0.35 (≥85% episode recall). **Maintenance** uses `maintenance_preset()` (≥90% event coverage, least minutes high).
- **Coverage:** Default preset uses `min_run=10`, `warmup=5`. Use `--max-coverage` for more (shorter) blocks; use `--no-coverage` for stricter blocks (min_run=20, warmup=10). Use **`--ultra-coverage`** (min_run=1, warmup=0) when maximizing **“episodes with a prediction in the window before”** is the priority: every run of 1+ minutes gets a block, so more fault episodes will have at least one prediction in the 6h/24h/3d before. **Trade-off:** features on 1‑minute runs are noisier (rolling windows use at most 1 running minute; buffer still gives 60 min of context). Use the same preset for train and predict (e.g. train and predict both with `--ultra-coverage`).

**Fundamental improvements when 24h/3d “high risk” still misses too many faults:**

- **Combine horizons:** Treat “high risk” as **high if (risk_6h == high) OR (risk_24h == high) OR (risk_aggregate == high) OR (risk_ttf == high)** so you catch more events (at the cost of more false alarms). Implement this in your derating logic when consuming `predictions.csv`.
- **Fault episodes (done):** Right now every *minute* with Leallas_zav/Gen_trip is a “fault event”, so one incident can count as hundreds of events. For 24h/3d you can instead label **one target per run segment** (e.g. “fault in next 24h” = 1 if this run has any fault in the next 24h). That reduces label noise and can improve the model; it would require a new label function and retrain.
- **Domain features (done):** Key signals get rate-of-change (roc_15m/30m/60m) and 2h window; see `FeatureConfig.key_signal_substrings`.
- **Different horizons:** Try training an extra model for **12h** as a middle ground between 6h and 24h, or use **risk_ttf** (minutes to fault) as the main short-term signal instead of P(fault in 6h).

### 9. Next steps

- Retrain periodically as new SCADA data arrives (fault and/or `--maintenance`).
- Use `review_thresholds.py` (and `--maintenance`) to tune risk thresholds before using predictions in operations.
- Use `streamlit run src/view_predictions_app.py` to inspect predictions by minute and day.
- Run `python scripts/build_report_docx.py` to generate the Hungarian Word report (`Elorejelzes_jelentes.docx`) from current results.
