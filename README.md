# pygifi: Multivariate Analysis with Optimal Scaling 🚀

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-green.svg)](https://opensource.org/licenses/GPL-3.0)
[![Tests](https://img.shields.io/badge/tests-149%20passed-brightgreen.svg)]()
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Welcome to **pygifi**! 🎉 This is a **Python port** of the legendary R [Gifi library](https://cran.r-project.org/package=Gifi) by Mair, De Leeuw, and Groenen. If you're into multivariate analysis, optimal scaling, or handling categorical data like a pro, you've come to the right place. Think of it as bringing the magic of R's Gifi to Python's ecosystem – no more switching tools! 📈

## 🌟 What's Inside This Project?

pygifi is your go-to library for **optimal scaling techniques** in multivariate data analysis. It's perfect for researchers, data scientists, and anyone dealing with mixed data types (categorical, ordinal, metric). Here's what you can do:

- **Homals (Homogeneity Analysis)**: Analyze categorical data with optimal scaling. Great for correspondence analysis on mixed variables. 📊
- **Princals (Principal Component Analysis with Optimal Scaling)**: Extend PCA to handle categorical/ordinal data seamlessly.
- **Morals (Multiple Optimal Scaling)**: For regression-like problems with optimal transformations.
- **Core Transformations**: Build custom optimal scaling with `gifi_transform`, isotone regression, and spline bases.
- **Built-in Datasets**: Test with real-world data like ABC, Galo, Hartigan, and more – no need to hunt for examples! 🗂️
- **Scikit-Learn Compatible**: Fits right into your ML pipelines with `fit()` and `transform()` methods.
- **Robust Handling**: Deals with missing data, ties, and various measurement levels (nominal, ordinal, metric).

Whether you're exploring data patterns, reducing dimensions, or preprocessing for machine learning, pygifi has you covered. It's like scikit-learn but specialized for categorical wizardry! 🧙‍♂️

## 📦 Installation: Get Started in Minutes!

Ready to dive in? Follow these steps – it's easier than brewing coffee! ☕

### Prerequisites
- **Python 3.8+**: Grab it from [python.org](https://www.python.org/downloads/) if you haven't.
- **Git**: For cloning the repo.
- **Optional**: A virtual environment (e.g., `venv` or `conda`) to keep things tidy.

### Step-by-Step Setup
1. **Clone the Repo** 📥  
   ```
   git clone https://github.com/your-username/pygifi.git  # Replace with your actual repo URL
   cd pygifi
   ```

2. **Create a Virtual Environment** (Recommended) 🏠  
   ```
   python -m venv pygifi-env
   source pygifi-env/bin/activate  # Windows: pygifi-env\Scripts\activate
   ```

3. **Install Dependencies** 📚  
   ```
   pip install numpy scipy pandas scikit-learn matplotlib
   ```

4. **Install pygifi** 🔧  
   ```
   pip install -e .
   ```
   Boom! You're all set. The `-e` flag means "editable" – any changes you make are instantly reflected.

### Quick Test: Is It Working? ✅
Run this in your terminal:
```
python -c "import pygifi; print('🎉 pygifi is ready!')"
```
If you see the emoji, you're golden. No errors? Great!

## 🚀 Quick Start: Your First Analysis

Let's analyze some data! Here's a simple example using the built-in ABC dataset. Copy-paste this into a Python script or Jupyter notebook:

```python
import pygifi
import pandas as pd

# Load a fun dataset (ABC is a classic for testing)
df = pygifi.get_dataset('ABC')
print("Dataset preview:")
print(df.head())

# Fit a Homals model for 2 dimensions
model = pygifi.Homals(ndim=2, levels='nominal')  # Treat all as categorical
model.fit(df)

# Check results
print("Object scores (first 5 rows):")
print(model.result_['objectscores'][:5])
print(f"Loss value: {model.result_['f']:.4f}")
print("Eigenvalues:", model.result_['evals'][:2])

# Plot if you want (requires matplotlib)
import matplotlib.pyplot as plt
plt.scatter(model.result_['objectscores'][:, 0], model.result_['objectscores'][:, 1])
plt.title("Homals Object Scores")
plt.show()
```

Output might look like:
```
Dataset preview:
   A  B  C
1  1  1  1
2  1  1  2
...
Object scores (first 5 rows):
[[-3.008  4.323]
 [-0.106 -1.392]
 ...]
Loss value: 0.5900
Eigenvalues: [1.234, 0.987]
```

```

Tada! You've just performed optimal scaling. Experiment with `levels='ordinal'` or try `Princals()` for PCA vibes. 📉

### ⚠️ A Note on Out-of-Sample Predictions
Because `pygifi` relies heavily on R's core Alternating Least Squares (ALS) and B-spline optimal scaling logic, it calculates optimal component knots *strictly based on the entire provided dataset structure*. 

**Out-of-sample prediction natively (like `model.transform(X_test)`) is currently not supported in this Python port.** 
If you project completely unseen data through the spline knot framework without re-fitting it, you will encounter a `NotImplementedError`. Re-fitting your model or utilizing cross-validation surrogate loss (like `cv_morals`) is required when testing split sample sets!

## 📁 Project Structure: What's What?

Confused by the folders? No worries – here's a breakdown with emojis for fun! Each file/folder is explained so you know exactly what's inside. Think of this as your treasure map. 🗺️

| Folder/File | Description | Emoji |
|-------------|-------------|-------|
| **`pygifi/`** | The heart of the library! Contains all the Python modules. | ❤️ |
| &nbsp;&nbsp;&nbsp;&nbsp;`__init__.py` | Main entry point – imports everything you need (Homals, Princals, etc.). | 🚪 |
| &nbsp;&nbsp;&nbsp;&nbsp;`homals.py` | Homogeneity Analysis class – your go-to for categorical PCA. | 📊 |
| &nbsp;&nbsp;&nbsp;&nbsp;`princals.py` | Principal Component Analysis with optimal scaling. | 🔍 |
| &nbsp;&nbsp;&nbsp;&nbsp;`morals.py` | Multiple Optimal Scaling for regression tasks. | 📈 |
| &nbsp;&nbsp;&nbsp;&nbsp;`plot.py` | Plotting utilities (e.g., for visualizing results). | 🎨 |
| &nbsp;&nbsp;&nbsp;&nbsp;`datasets.py` | Loads built-in datasets like ABC, Galo, etc. Perfect for testing! | 📚 |
| &nbsp;&nbsp;&nbsp;&nbsp;`cv.py` | Cross-validation tools for MORALS. | 🔄 |
| &nbsp;&nbsp;&nbsp;&nbsp;`_engine.py` | Core ALS (Alternating Least Squares) engine – the brain behind the algorithms. | 🧠 |
| &nbsp;&nbsp;&nbsp;&nbsp;`_isotone.py` | Isotonic regression and PAVA (Pool Adjacent Violators Algorithm). | 📏 |
| &nbsp;&nbsp;&nbsp;&nbsp;`_coding.py` | Encoding/decoding for categorical data (like R's factor handling). | 🔢 |
| &nbsp;&nbsp;&nbsp;&nbsp;`_linalg.py` | Linear algebra helpers (e.g., Gram-Schmidt, least squares). | ➗ |
| &nbsp;&nbsp;&nbsp;&nbsp;`_splines.py` | Spline basis functions and knots for smooth transformations. | 🌊 |
| &nbsp;&nbsp;&nbsp;&nbsp;`_prepspline.py` | Prepares spline parameters from measurement levels. | 🛠️ |
| &nbsp;&nbsp;&nbsp;&nbsp;`_structures.py` | Data structures for Gifi objects (like R's lists). | 🏗️ |
| &nbsp;&nbsp;&nbsp;&nbsp;`_utilities.py` | Handy utilities: centering, normalizing, correlations, etc. | 🧰 |
| &nbsp;&nbsp;&nbsp;&nbsp;`data/` | Folder with CSV files of sample datasets (ABC.csv, galo.csv, etc.). | 📁 |
| **`tests/`** | Test suite – 149 tests to ensure everything works. Run with `pytest`. | 🧪 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_*.py` | Individual test files for each module (e.g., test_homals.py). | 📝 |
| &nbsp;&nbsp;&nbsp;&nbsp;`fixtures/` | Test fixtures, including R scripts to generate reference data. | 🔧 |
| **`pyproject.toml`** | Project config – defines dependencies, version, and build settings. | ⚙️ |
| **`README.md`** | This file! Your guide to all things pygifi. | 📖 |
| **`LICENSE`** | GPL-3.0-or-later license (same as the original R package). | 📜 |
| **`CHANGELOG.md`** | What's new in each version. | 📅 |

Pro Tip: Start with `pygifi/__init__.py` to see what's available, then dive into `homals.py` for your first analysis. The `_` prefixed files are internal helpers – you won't need them directly, but they're the magic under the hood! ✨

## 🧪 Testing: Make Sure It Works

We love tests! To run the full suite:
```
pip install pytest
pytest
```
Expect 149 passing tests in ~2-3 minutes. If any fail, check your Python version or dependencies.

For a quick smoke test:
```python
import pygifi
df = pygifi.get_dataset('small')
model = pygifi.Homals(ndim=1)
model.fit(df)
print("Success!" if model.result_['f'] < 1 else "Hmm...")
```

## 🔍 Comparison with R Gifi

Curious how it stacks up? `pygifi` is a deterministic parity port: we map the identical knot logic, tie handling, and missing-value distribution techniques from the CRAN `.R` engine. 

Results strictly match R within tight floating-point convergence precision bounds! Install R Gifi and compare them using an identical global mathematical objective. 📏

## 🤝 Contributing & Issues

Found a bug? Want a feature? Open an issue on GitHub! Contributions welcome – fork, code, and PR. Let's make pygifi even better! 🌟

## 📄 License

This project is licensed under GPL-3.0-or-later, just like the original R Gifi. Free as in freedom! 🆓

---

Happy analyzing! If you get stuck, check the docstrings (`help(pygifi.Homals)`) or ping us. Let's unlock the power of optimal scaling together! 🚀📊