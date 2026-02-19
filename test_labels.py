"""Test maintenance event detection and labeling (refined: shutdown fault + min 30min downtime)."""
from pathlib import Path
from src.load_data import build_base_frame
from src.label_events import add_maintenance_labels

print("Loading SCADA data...")
df = build_base_frame(Path("Data"))

print("Adding maintenance labels (run→stop, not Oper_off, downtime >= 5 min)...")
df = add_maintenance_labels(df, run_col="Gen_cb_cld", oper_off_col="Oper_off")

print("\nMaintenance event summary (refined):")
events = df["maintenance_event"].sum()
print(f"  Total maintenance events: {events}")

if events > 0:
    event_times = df.index[df["maintenance_event"]]
    print(f"\nFirst 5 maintenance events:")
    for i, ts in enumerate(event_times[:5], 1):
        print(f"  {i}. {ts}")

print("\nLabel statistics:")
print(f"  y_maint_24h positive labels: {df['y_maint_24h'].sum()}")
print(f"  y_maint_7d positive labels:  {df['y_maint_7d'].sum()}")
print(f"  y_maint_30d positive labels: {df['y_maint_30d'].sum()}")
print(f"  y_maint_24h positive rate:   {df['y_maint_24h'].mean():.4f}")
print(f"  y_maint_7d positive rate:   {df['y_maint_7d'].mean():.4f}")
print(f"  y_maint_30d positive rate:   {df['y_maint_30d'].mean():.4f}")

print("\n[OK] Label validation completed")
