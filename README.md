# pygifi — Python Port of R's Gifi Library 🐍📊

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-green.svg)](https://opensource.org/licenses/GPL-3.0)
[![CI](https://github.com/honey-of-blood/pyGifi/actions/workflows/ci.yml/badge.svg)](https://github.com/honey-of-blood/pyGifi/actions/workflows/ci.yml)

**pygifi** is a Python port of the R [Gifi package](https://cran.r-project.org/package=Gifi) by Mair, De Leeuw, and Groenen.  
It brings **multivariate analysis with optimal scaling** to Python — handling categorical, ordinal, and mixed-type data natively in a scikit-learn compatible API.

---

## ✨ What Does This Library Do?

Gifi methods are a family of algorithms that find the best numerical representation of any kind of data — even if it's categorical or ordinal — so you can apply linear methods like PCA or regression to it. Each variable is **optimally transformed** to maximize structure.

| Method | Class | What it does |
|--------|-------|-------------|
| Homogeneity Analysis | `Homals` | Like Multiple Correspondence Analysis (MCA). Finds groups and patterns in categorical data. |
| Optimal Scaling PCA | `Princals` | Like PCA but works on any mix of nominal, ordinal, and numeric variables. |
| Monotone Regression | `Morals` | Like linear regression but the predictors/response are optimally transformed (monotone). |
| Correlational Analysis | `Corals` | Maximizes correlation between two sets of variables. |
| Canonical Analysis | `Canals` | Canonical correlation with optimal scaling. |
| Discriminant Analysis | `Criminals` | Nonlinear discriminant analysis. |
| Multiset Analysis | `Overals` | Generalizes Homals/Corals to multiple sets. |
| Primal Analysis | `Primals` | Regression with metric response. |
| Additive Analysis | `Addals` | Additive models with optimal scaling. |
| Missing Data Imputation | `GifiIterativeImputer` | Iterative imputation oriented to the Gifi framework. |

**Additional utilities:**
- `pygifi.plot()` — unified plot dispatcher (loading plots, biplots, transformation plots, object score plots)
- `pygifi.get_dataset()` — 12 built-in classic datasets
- `pygifi.make_numeric()` / `encode()` / `decode()` — categorical coding utilities
- `pygifi.cv_morals()` — cross-validation for Morals
- `pygifi.knots_gifi()` — B-spline knot placement (matches R exactly)

---

## 📦 Installation

### Requirements

- Python **3.9 or above**
- pip

### Install from source (recommended until PyPI release)

```bash
# 1. Clone the repository
git clone https://github.com/honey-of-blood/pyGifi.git
cd pyGifi

# 2. (Optional but recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install the package with all dependencies
pip install -e .
```

The `-e` flag installs in **editable mode**, so any local file changes are reflected immediately without reinstalling.

### Dependencies installed automatically

| Package | Purpose |
|---------|---------|
| `numpy >= 1.21` | Numerical computation |
| `scipy >= 1.7` | Linear algebra, splines, NNLS |
| `pandas >= 1.3` | DataFrame handling |
| `scikit-learn >= 1.0` | Base estimator interface |
| `matplotlib >= 3.4` | Plotting |

### Optional: Numba acceleration

For faster PAVA (isotone regression) on large datasets:

```bash
pip install -e ".[accelerate]"
```

### Verify installation

```bash
python -c "import pygifi; print(pygifi.__version__)"
```

---

## 🚀 Quick Start

### 1. Homogeneity Analysis (Homals)

Discover structure in categorical data:

```python
import pygifi

# Load a built-in dataset (11 classic datasets available)
df = pygifi.get_dataset('ABC')

# Fit Homals — 2 dimensions, treat all columns as nominal
model = pygifi.Homals(ndim=2, levels='nominal')
model.fit(df)

# Inspect results
print(model)                              # summary
print(model.result_['objectscores'][:5]) # row coordinates
print(model.result_['evals'])            # eigenvalues

# Plot
pygifi.plot(model, plot_type='objplot')  # object scores
pygifi.plot(model, plot_type='loadplot') # variable loadings
```

### 2. Optimal Scaling PCA (Princals)

Like PCA for mixed-type data:

```python
import pygifi

df = pygifi.get_dataset('galo')

model = pygifi.Princals(ndim=2, levels=['nominal', 'ordinal', 'nominal', 'ordinal'])
model.fit(df)
print(model)
pygifi.plot(model, plot_type='biplot')
```

### 3. Monotone Regression (Morals)

Regression where variables are monotonically transformed:

```python
import pygifi
import pandas as pd

df = pygifi.get_dataset('neumann')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = pygifi.Morals(xdegrees=2, ydegrees=2, xordinal=True, yordinal=True)
model.fit(X, y)
print(model)                        # SMC, loss, iterations
pygifi.plot(model, plot_type='transplot')
```

### 4. Available Built-in Datasets

```python
import pygifi

# See all available datasets
datasets = ['ABC', 'galo', 'hartigan', 'neumann', 'mammals',
            'roskam', 'senate07', 'gubell', 'house', 'sleeping',
            'small', 'WilPat2']

df = pygifi.get_dataset('hartigan')
print(df.head())
```

### 5. Categorical Encoding Utilities

```python
import pygifi
import pandas as pd

df = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': ['S', 'L', 'M']})

# Encode strings to integer codes
encoded, mapping = pygifi.categorical_encode(df)

# Decode back to original
decoded = pygifi.categorical_decode(encoded, mapping)
```

---

## ⚠️ Important Limitations

- **Out-of-sample prediction is not supported.** The optimal transformations are computed on the entire dataset at once. Calling `model.transform(X_test)` on unseen data will raise `NotImplementedError`. Use `cv_morals()` for cross-validation instead.
- Results match R's Gifi within floating-point tolerance. Large datasets may show minor differences due to RNG initialization differences.

---

## ⚖️ Validation Against R's Gifi (How Comparison Works)

This project includes a full **validation suite** to verify that Python results match R's output exactly (within tolerance `1e-3`).

### Step-by-step: How to run the comparison yourself

#### 1. Prepare your dataset

Place your CSV dataset in:
```
validation/datasets/
```

For example: `validation/datasets/my_data.csv`

The CSV should be comma-separated with a header row. Column types (nominal, ordinal) are handled by the Python and R scripts.

#### 2. Run the R script

Install R's Gifi package first:
```r
install.packages("Gifi")
```

Then run the R script (which reads your dataset and writes results):
```bash
Rscript validation/r_scripts/run_gifi.R
```

This produces CSV files in `validation/results/` prefixed with `r_` (e.g., `r_homals_my_data.csv`).

#### 3. Run the Python script

```bash
python validation/python_scripts/run_pygifi.py
```

This produces CSV files in `validation/results/` prefixed with `py_` (e.g., `py_homals_my_data.csv`).

#### 4. Compare results

```bash
python validation/compare/compare_results.py
```

This prints a report comparing R and Python outputs element-wise and flags any differences exceeding the tolerance.

#### 5. Generate a full validation report

```bash
python validation/report.py
```

> Results CSV files for `bike_dataset` and `car_dataset` are already pre-computed in `validation/results/` so you can see example outputs immediately.

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (excluding slow R-parity tests)
pytest tests/ --ignore=tests/test_parity.py -v

# Run with coverage
pytest tests/ --ignore=tests/test_parity.py --cov=pygifi --cov-report=term-missing
```

---

## 📁 Project Structure — Explained for Beginners

```
pyGifi/
│
├── pygifi/                         ← The actual Python library (install this)
│   ├── __init__.py                 ← Entry point: exposes all public classes/functions
│   │
│   ├── models/                     ← One file per Gifi algorithm
│   │   ├── homals.py               ← Homals: homogeneity analysis (MCA-style)
│   │   ├── princals.py             ← Princals: optimal scaling PCA
│   │   ├── morals.py               ← Morals: monotone regression
│   │   ├── corals.py               ← Corals: correlation analysis
│   │   ├── canals.py               ← Canals: canonical analysis
│   │   ├── criminals.py            ← Criminals: discriminant analysis
│   │   ├── overals.py              ← Overals: multi-set analysis
│   │   ├── primals.py              ← Primals: metric regression
│   │   ├── addals.py               ← Addals: additive analysis
│   │   └── impute.py               ← GifiIterativeImputer: missing value imputation
│   │
│   ├── core/                       ← Engine internals (you don't need to touch these)
│   │   ├── engine.py               ← Main ALS loop (gifi_engine) + transformation routing
│   │   ├── structures.py           ← Data structure builders (make_gifi, make_x_gifi)
│   │   ├── linalg.py               ← Linear algebra helpers (Gram-Schmidt, least squares)
│   │   └── cv.py                   ← Cross-validation: cv_morals()
│   │
│   ├── utils/                      ← Low-level utilities (internal helpers)
│   │   ├── _cone.py                ← Cone projection router (project_cone) — ALS H-update
│   │   ├── isotone.py              ← PAVA, Dykstra, monotone regression algorithms
│   │   ├── splines.py              ← B-spline basis construction (matches R's splineDesign)
│   │   ├── coding.py               ← make_numeric, encode/decode, categorical coding
│   │   ├── utilities.py            ← center, normalize, reshape, GS-orthogonalize helpers
│   │   └── prepspline.py           ← Spline knot pre-processing utilities
│   │
│   ├── visualization/              ← All plotting code
│   │   └── plot.py                 ← plot(), plot_homals, plot_princals, plot_morals, biplot
│   │
│   └── data/                       ← Built-in datasets (CSV files, loaded by get_dataset())
│       ├── ABC.csv, galo.csv, hartigan.csv, neumann.csv, mammals.csv ...
│
├── tests/                          ← Automated test suite
│   ├── test_homals.py              ← Tests for Homals model
│   ├── test_princals.py            ← Tests for Princals model
│   ├── test_morals.py              ← Tests for Morals model
│   ├── test_engine.py              ← Tests for the core ALS engine
│   ├── test_isotone.py             ← Tests for PAVA / monotone regression
│   ├── test_coding.py              ← Tests for encoding utilities
│   ├── test_splines.py             ← Tests for B-spline construction
│   ├── test_linalg.py              ← Tests for linear algebra helpers
│   ├── test_structures.py          ← Tests for data structure builders
│   ├── test_parity.py              ← (Slow) R vs Python parity tests — requires R installed
│   ├── test_plot.py / test_plot2.py / test_plot_unified.py ← Plot smoke tests
│   └── fixtures/                   ← Pre-computed R output files used as ground truth
│       ├── homals_hartigan.json    ← R Homals output on hartigan dataset
│       ├── princals_abc.json       ← R Princals output on ABC dataset
│       ├── morals_neumann.json     ← R Morals output on neumann dataset
│       └── ...                     ← Other reference outputs
│
├── validation/                     ← Cross-validation: Python vs R comparison suite
│   ├── datasets/                   ← Put YOUR datasets here for comparison
│   │   ├── bike_dataset.csv        ← Example dataset (pre-included)
│   │   ├── car_dataset.csv         ← Example dataset (pre-included)
│   │   └── processed/              ← Cleaned versions of datasets after preprocessing
│   ├── python_scripts/
│   │   └── run_pygifi.py           ← Runs pygifi on the datasets, writes py_*.csv results
│   ├── r_scripts/
│   │   └── run_gifi.R              ← Runs R Gifi on the datasets, writes r_*.csv results
│   ├── compare/
│   │   └── compare_results.py      ← Reads py_*.csv and r_*.csv and checks they match
│   ├── results/                    ← Output CSVs from both Python and R (auto-generated)
│   │   ├── py_homals_bike_dataset.csv
│   │   ├── r_homals_bike_dataset.csv
│   │   └── ...
│   ├── preprocess_datasets.py      ← Cleans raw datasets before running comparisons
│   └── report.py                   ← Generates a combined validation report
│
├── docs/                           ← Documentation and theory notebooks
│   ├── tutorial.md                 ← Quickstart tutorial
│   ├── api.md                      ← Full API reference
│   ├── theory.md                   ← Mathematical background on Gifi methods
│   ├── gifi_theory.ipynb           ← Jupyter notebook: theory deep-dive
│   ├── homals_tutorial.ipynb       ← Jupyter notebook: Homals walkthrough
│   ├── princals_tutorial.ipynb     ← Jupyter notebook: Princals walkthrough
│   └── morals_tutorial.ipynb       ← Jupyter notebook: Morals walkthrough
│
├── examples/                       ← Standalone worked examples
│   ├── homals_example.ipynb
│   ├── princals_example.ipynb
│   └── morals_example.ipynb
│
├── pyproject.toml                  ← Package configuration: dependencies, version, metadata
├── setup.py                        ← Legacy build script (kept for compatibility)
├── MANIFEST.in                     ← Tells pip which non-Python files to include
├── CHANGELOG.md                    ← Version history and what changed
├── LICENSE                         ← GPL-3.0 license
└── README.md                       ← This file
```

---

## 📄 License

GPL-3.0-or-later — same as the original R Gifi package.

## 🙏 Credits

Original R Gifi package by Patrick Mair, Jan de Leeuw, and Patrick Groenen. Python port developed for research reproducibility.