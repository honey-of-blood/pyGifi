# ==============================================================
# PYGIFI — BIKE BUYERS DATASET
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygifi
from pygifi import Homals, Princals, Morals

print(f"pygifi version : {pygifi.__version__}")
print(f"numpy          : {np.__version__}")
print(f"pandas         : {pd.__version__}")

# ==============================================================
# HELPERS
# ==============================================================

def separator(title):
    print("\n" + "="*54)
    print(f"  {title}")
    print("="*54)


def print_matrix(arr, row_labels=None, col_labels=None, indent="  "):
    arr = np.atleast_2d(np.array(arr))
    nrow, ncol = arr.shape

    col_labels = col_labels or [f"[,{i+1}]" for i in range(ncol)]
    row_labels = row_labels or [f"R{i+1}" for i in range(nrow)]

    hdr = indent + f"{'':18s}" + "".join(f"{c:>13s}" for c in col_labels)
    print(hdr)

    for r, row in zip(row_labels, arr):
        vals = "".join(f"{v:13.7f}" for v in row)
        print(f"{indent}{r:18s}{vals}")


def get_result(model):
    return model.result_ if hasattr(model, "result_") else vars(model)


# ==============================================================
# 1. LOAD DATASET
# ==============================================================

separator("DATASET LOADING")

DATA_PATH = r"E:/Anti projects/pyGifi/PyGifi2/PyGifi2/datasets/processed/bike_dataset.csv"

df = pd.read_csv(DATA_PATH)

# fix column names (replace spaces with dots)
df.columns = df.columns.str.replace(" ", ".")

print("Shape :", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# convert to categorical
for col in df.columns:
    df[col] = df[col].astype("category")

# predictor columns
COLS = [
    "Marital.Status",
    "Gender",
    "Income",
    "Children",
    "Education",
    "Occupation",
    "Home.Owner",
    "Cars",
    "Commute.Distance",
    "Region",
    "Age"
]

print("\nColumn level counts:")
for col in COLS:
    lvls = list(df[col].cat.categories)
    print(f"  {col:18s}: {len(lvls)} levels {lvls}")

# predictors
df_coded = df[COLS]

# response variable
income = (df["Purchased.Bike"] == "Yes").astype(float)

print(
    f"\nIncome: {int((income==0).sum())} zeros, "
    f"{int((income==1).sum())} ones "
    f"({income.mean()*100:.1f}% positive)"
)

# ==============================================================
# 2. HOMALS
# ==============================================================

separator("HOMALS")

h_fit = Homals(ndim=2).fit(df_coded)
r_h = get_result(h_fit)

loss_h = r_h.get("f")
evals_h = np.array(r_h.get("evals", [])).flatten()
ntel_h = r_h.get("ntel")

print(f"\nLoss (f)          : {loss_h:.9f}")

if len(evals_h):
    total_h = evals_h.sum()
    vaf_h = evals_h / total_h * 100
    cumvaf_h = np.cumsum(vaf_h)

    print(f"Eigenvalue D1     : {evals_h[0]:.9f}")
    print(f"Eigenvalue D2     : {evals_h[1]:.9f}")
    print(f"Total eigenvalues : {len(evals_h)}")
    print(f"VAF D1            : {vaf_h[0]:.6f}")
    print(f"Cumulative VAF    : {cumvaf_h[1]:.6f}")

print(f"\nntel              : {ntel_h}")

# object scores
scores_h = np.array(h_fit.transform(df_coded))

print("\nObject Scores (first 5 rows):")

print_matrix(
    scores_h[:5],
    row_labels=[str(i+1) for i in range(5)],
    col_labels=["D1", "D2"]
)

# ==============================================================
# 3. PRINCALS (Nominal)
# ==============================================================

separator("PRINCALS (Nominal)")

pn_fit = Princals(ndim=2).fit(df_coded)
r_pn = get_result(pn_fit)

loss_pn = r_pn.get("f")
evals_pn = np.array(r_pn.get("evals", [])).flatten()
load_pn = np.array(r_pn.get("loadings"))
ntel_pn = r_pn.get("ntel")

print(f"\nLoss (f)          : {loss_pn:.9f}")
print(f"Eigenvalue D1     : {evals_pn[0]:.9f}")
print(f"Eigenvalue D2     : {evals_pn[1]:.9f}")
print(f"ntel              : {ntel_pn}")

print("\nLoadings:")

print_matrix(load_pn, row_labels=COLS, col_labels=["D1", "D2"])

# ==============================================================
# 4. PRINCALS (Ordinal)
# ==============================================================

separator("PRINCALS (Ordinal)")

po_fit = Princals(ndim=2, levels="ordinal").fit(df_coded)
r_po = get_result(po_fit)

loss_po = r_po.get("f")
evals_po = np.array(r_po.get("evals", [])).flatten()
load_po = np.array(r_po.get("loadings"))

print(f"\nLoss (f)          : {loss_po:.9f}")
print(f"Eigenvalue D1     : {evals_po[0]:.9f}")

print("\nLoadings:")

print_matrix(load_po, row_labels=COLS, col_labels=["D1", "D2"])

# ==============================================================
# 5. MORALS
# ==============================================================

separator("MORALS")

m_fit = Morals().fit(df_coded, income)
r_m = get_result(m_fit)

smc_m = r_m.get("smc")
loss_m = r_m.get("f")
beta_m = np.array(r_m.get("beta", [])).flatten()
ntel_m = r_m.get("ntel")

print(f"\nSMC (R²)          : {smc_m:.9f}")
print(f"Loss (f)          : {loss_m:.9f}")
print(f"ntel              : {ntel_m}")

print("\nBeta coefficients:")

for var, b in zip(COLS, beta_m):
    print(f"  {var:18s}{b:12.6f}")

# ==============================================================
# 6. VISUALIZATION
# ==============================================================

separator("VISUALIZATION")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

fig.suptitle(
    "pygifi Analysis — Bike Buyers Dataset",
    fontsize=13,
    fontweight="bold"
)

# HOMALS scores
ax = axes[0]

ax.scatter(scores_h[:, 0], scores_h[:, 1], alpha=0.4, s=15)

ax.axhline(0)
ax.axvline(0)

ax.set_title("HOMALS Object Scores")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")

# PRINCALS loadings
ax = axes[1]

for i, var in enumerate(COLS):

    ax.annotate(
        "",
        xy=(load_pn[i, 0], load_pn[i, 1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->")
    )

    ax.text(
        load_pn[i, 0] * 1.15,
        load_pn[i, 1] * 1.15,
        var,
        fontsize=8
    )

ax.set_title("PRINCALS Loadings")
ax.set_xlabel("Dim1")
ax.set_ylabel("Dim2")

# MORALS beta
ax = axes[2]

ax.barh(COLS, beta_m)

ax.set_title("MORALS Beta Coefficients")
ax.set_xlabel("Beta")

plt.tight_layout()

plt.savefig("pygifi_bike_analysis.png", dpi=150)

print("Plot saved -> pygifi_bike_analysis.png")

print("\n" + "="*54)
print("Script completed")
print("="*54)