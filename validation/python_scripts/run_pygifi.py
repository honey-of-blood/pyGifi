"""
run_pygifi.py — Generate per-parameter result CSVs from PyGifi for Princals only.

Exports:
  1. Category Quantifications per variable
  2. The full PyGifi Transformed Dataset
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure pygifi can be imported when run from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # PyGifi2/PyGifi2/
sys.path.insert(0, _ROOT)

from pygifi import Princals

DATASETS_DIR = os.path.join(_HERE, "..", "datasets", "processed")
RESULTS_DIR  = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

datasets = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv") and "transformed" not in f])

def _save_df(df, path):
    """Save a pandas DataFrame to CSV."""
    df.to_csv(path)
    print(f"    -> {os.path.basename(path)}  {df.shape}")

for ds in datasets:
    prefix = ds.replace(".csv", "")
    data = pd.read_csv(os.path.join(DATASETS_DIR, ds))

    # Drop unnamed index columns
    data.drop(columns=[c for c in data.columns if "Unnamed" in c], inplace=True)

    # All columns will be converted to categorical
    for col in data.columns:
        data[col] = data[col].astype("category")

    print(f"\n[{prefix}]")

    # ─────────────────────────────────────────────────────────────────
    # PRINCALS
    # ─────────────────────────────────────────────────────────────────
    print("  Fitting PyGifi PRINCALS ...")
    p = Princals(ndim=2)
    p.fit(data)
    r = p.result_
    base = f"py_princals_{prefix}"

    quantifications = r["quantifications"]

    # 1. Export Category Quantifications
    for col, q in zip(data.columns, quantifications):
        # Match R's default replacement of spaces to periods
        safe = col.replace(" ", ".").replace("/", ".")
        categories = data[col].cat.categories
        
        # We only care about dimension 1 mapping as standard for report, but export all.
        q_df = pd.DataFrame(q, index=categories, columns=[f"dim{i+1}" for i in range(q.shape[1])])
        q_df.index.name = "Category"
        _save_df(q_df, os.path.join(RESULTS_DIR, f"{base}_quant_{safe}.csv"))

    # 2. Build and export the Transformed Dataset
    df_transformed = pd.DataFrame(p.result_["transform"], index=data.index, columns=data.columns)

    transformed_path = os.path.join(DATASETS_DIR, f"pygifi_transformed_{prefix}.csv")
    df_transformed.to_csv(transformed_path, index=False)
    print(f"    -> {os.path.basename(transformed_path)}  {df_transformed.shape} [Transformed Dataset]")

print("\nDone — all PyGifi Princals results exported.")
