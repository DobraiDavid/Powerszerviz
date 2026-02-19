"""Test feature building speed."""
import time
from pathlib import Path
from src.load_data import build_base_frame
from src.label_events import add_fault_horizon_labels
from src.features import build_features, FeatureConfig

print("Loading data...")
t0 = time.time()
df = build_base_frame(Path("Data"))
print(f"Data loading took {time.time() - t0:.1f} seconds")

print("Adding fault-horizon labels...")
t0 = time.time()
df = add_fault_horizon_labels(df, run_col="Gen_cb_cld")
print(f"Labeling took {time.time() - t0:.1f} seconds")

print("Building features...")
t0 = time.time()
config = FeatureConfig()
X, y_24h, y_7d, y_30d, mask_trn = build_features(df, config=config, run_col="Gen_cb_cld", label_prefix="fault")
print(f"Feature building took {time.time() - t0:.1f} seconds")
print(f"Features shape: {X.shape}")
print(f"Training mask covers {mask_trn.sum()} rows out of {len(mask_trn)}")
