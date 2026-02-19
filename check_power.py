"""Check power columns in the data."""
from pathlib import Path
from src.load_data import build_base_frame

df = build_base_frame(Path("Data"))

# Check for Gen_visz_telj
if "Gen_visz_telj" in df.columns:
    print(f"Gen_visz_telj found:")
    print(f"  min={df['Gen_visz_telj'].min():.2f}")
    print(f"  max={df['Gen_visz_telj'].max():.2f}")
    print(f"  mean={df['Gen_visz_telj'].mean():.2f}")
    print(f"  non-null count={df['Gen_visz_telj'].notna().sum()}")
else:
    print("Gen_visz_telj NOT found")
    print("\nSearching for power-related columns:")
    power_cols = [c for c in df.columns if 'telj' in c.lower() or 'power' in c.lower()]
    for col in power_cols[:10]:  # Show first 10
        if df[col].dtype in ['float64', 'int64']:
            print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}")
