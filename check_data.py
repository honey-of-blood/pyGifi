import pandas as pd
import numpy as np

DATA_PATH = "/media/bhaavesh/New Volume/studies/antigravity-projects/pyGifi/PyGifi2/car_pure_categorical_1000.csv"

# Python loading
df_py = pd.read_csv(DATA_PATH)
# R loading (approximate)
df_r_sim = pd.read_csv(DATA_PATH, na_values="")

print("--- Python Loading ---")
for col in df_py.columns:
    lvls = df_py[col].astype("category").cat.categories
    print(f"{col:15s}: {len(lvls)} levels. First 3: {list(lvls[:3])}")
    print(f"  Missing: {df_py[col].isna().sum()}")

print("\n--- R Simulation Loading (na_values='') ---")
for col in df_r_sim.columns:
    lvls = df_r_sim[col].astype("category").cat.categories
    print(f"{col:15s}: {len(lvls)} levels. First 3: {list(lvls[:3])}")
    print(f"  Missing: {df_r_sim[col].isna().sum()}")
