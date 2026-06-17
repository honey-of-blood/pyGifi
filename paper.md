---
title: 'pyGifi: A Python Library for Multivariate Analysis with Optimal Scaling'

tags:
  - Python
  - optimal scaling
  - multivariate analysis
  - categorical data
  - principal components
  - alternating least squares
  - isotone regression
  - multiple correspondence analysis

authors:
  - name: Author One
    orcid: 0000-0000-0000-0001
    affiliation: 1
  - name: Author Two
    orcid: 0000-0000-0000-0002
    affiliation: 1

affiliations:
  - name: Institution Name, Country
    index: 1

date: 17 June 2026
bibliography: paper.bib
---

# Summary

`pyGifi` is a Python library for multivariate analysis with optimal scaling. It is a faithful port of the R `Gifi` package [@mair2014gifi], which implements the unified Gifi system of nonlinear multivariate analysis originally developed by a collective of Dutch statisticians working under the pseudonym Albert Gifi [@gifi1990nonlinear]. The library provides a family of algorithms — including Homogeneity Analysis (`Homals`), Categorical Principal Components Analysis (`Princals`), and Monotone Regression (`Morals`), among others — that can jointly analyze data containing any mix of nominal (unordered categorical), ordinal (ordered categorical), and metric (continuous) variables by assigning them optimal numerical representations before fitting a linear model.

At the core of all algorithms is an **Alternating Least Squares (ALS)** engine that iteratively minimizes a loss function measuring the discrepancy between object scores in a low-dimensional space and the optimally scaled variables. Each variable's transformation is constrained to lie in a *cone* (a convex admissible set) defined by its measurement level: the column space of an indicator or B-spline basis for nominal and spline variables, a monotone isotone cone enforced by the Pool Adjacent Violators Algorithm (PAVA) for ordinal variables, or their intersection via Dykstra's alternating projection for ordinal-spline variables.

`pyGifi` exposes a scikit-learn-compatible API [@scikit-learn], includes 12 built-in classic datasets, a three-phase automated numerical validation suite against R, a C extension that reproduces R's Mersenne-Twister random number stream for exact parity testing, and visualization utilities for all supported model types.

# Statement of Need

Mixed-type data — datasets in which some columns are categorical, others are ordinal, and others are continuous — is ubiquitous in the social sciences, healthcare, market research, and survey analysis. Classical dimensionality reduction tools such as PCA [@jolliffe2002principal] and factor analysis assume all variables are metric, meaning they impose arbitrary linear distances on categorical codes (e.g., treating "Red = 1, Blue = 2, Green = 3" as if Blue is numerically halfway between Red and Green). This assumption is statistically invalid and can severely distort downstream analysis.

The Gifi system resolves this by treating the numerical representation of each variable as an unknown to be *optimized*: optimal scaling finds the category quantifications that maximize the variance explained by the model while strictly respecting the variable's measurement level. The practical consequence is that analysts can apply classical PCA-like decompositions to any dataset, regardless of column types, without making invalid metric assumptions.

While R users have access to the mature `Gifi` [@mair2014gifi] and `homals` [@de2009gifi] packages, and SPSS offers a proprietary CATPCA implementation, **no complete, actively-maintained Python implementation existed** prior to `pyGifi`. Existing Python alternatives are partial:

- `prince` provides Multiple Correspondence Analysis and Factor Analysis of Mixed Data, but does not implement the full Gifi algorithm family, does not support ordinal constraints, and has no B-spline basis support.
- `scikit-learn`'s `OrdinalEncoder` and `OneHotEncoder` encode categorical variables but do not optimize the encoding.
- `mca` implements a standalone MCA but is not maintained and lacks the broader Gifi model family.

`pyGifi` fills this gap by providing:
1. A complete, faithful Python port of the full Gifi algorithm family with validated numerical parity to the R reference implementation.
2. A scikit-learn-compatible API so that Gifi models integrate naturally into `Pipeline` and `GridSearchCV` workflows.
3. Open, well-documented source code enabling reproducibility of published results that previously required R.

The library is intended for researchers in statistics, psychometrics, social sciences, and data science practitioners who work with mixed-type or predominantly categorical datasets.

# Mathematical Background

## Optimal Scaling

Let $\mathbf{X}$ be a data matrix with $n$ rows (observations) and $p$ columns (variables), where columns may be of different measurement levels. For each variable $j$, define a *basis matrix* $\mathbf{G}_j$ ($n \times k_j$) that encodes the permissible transformations:

- **Nominal**: $\mathbf{G}_j$ is the one-hot indicator matrix of the categories.
- **Ordinal**: $\mathbf{G}_j$ is the indicator matrix, but the resulting transformation must be isotone (non-decreasing) with respect to the original order.
- **Metric / Spline**: $\mathbf{G}_j$ is a B-spline design matrix [@deboor1978practical] evaluated at the data values.

The Gifi loss function (stress) is:

$$\text{stress} = \frac{1}{J \cdot d} \sum_{j=1}^{J} \left\| \mathbf{X} - \mathbf{H}_j \mathbf{A}_j \right\|_F^2$$

where $\mathbf{X}$ ($n \times d$) contains the *object scores* (latent coordinates), $\mathbf{H}_j$ ($n \times r_j$) is the optimal transformation of variable $j$, $\mathbf{A}_j$ ($r_j \times d$) is the weight matrix, $d$ is the number of dimensions, and $J$ is the number of active variable sets.

## Alternating Least Squares

`gifi_engine` minimizes the stress by alternating between two update steps until convergence [@young1981quantitative]:

**Step 1 — Model update (fix $\mathbf{H}_j$, update $\mathbf{X}$).**
Accumulate the predicted scores across all variable sets and re-orthogonalize:
$$\mathbf{X} \leftarrow \text{GS}\!\left(\text{center}\!\left(\sum_i \sum_{j \in S_i} \mathbf{H}_j \mathbf{A}_j\right)\right)$$
where $\text{GS}(\cdot)$ denotes column-pivoted Modified Gram-Schmidt orthogonalization [@golub2013matrix].

**Step 2 — Optimal scaling update (fix $\mathbf{X}$, update $\mathbf{H}_j$).**
For each variable $j$, compute the majorization target:
$$\tilde{\mathbf{h}}_j = \mathbf{h}_j + \frac{1}{\kappa} \mathbf{r} \mathbf{A}_j^\top$$
where $\kappa = \lambda_{\max}(\mathbf{A}_j^\top \mathbf{A}_j)$ is the Lipschitz constant (the largest eigenvalue of the set-level coefficient matrix) and $\mathbf{r}$ is the residual matrix. Then project $\tilde{\mathbf{h}}_j$ onto the admissible cone $\mathcal{K}_j$:

$$\mathbf{H}_j \leftarrow \Pi_{\mathcal{K}_j}(\tilde{\mathbf{h}}_j)$$

The cone projection $\Pi_{\mathcal{K}_j}$ routes to one of:

- **Subspace** (`'s'`): ordinary least squares onto $\text{col}(\mathbf{G}_j)$, solved via Modified Gram-Schmidt.
- **Isotone categorical** (`'c'`): PAVA applied to grouped category means of the target, enforcing the monotone ordering of category quantifications.
- **Dykstra** (`'i'`): alternating projections onto $\text{col}(\mathbf{G}_j)$ and the isotone cone simultaneously, for ordinal spline/polynomial variables [@dykstra1983algorithm].

## Pool Adjacent Violators Algorithm (PAVA)

For ordinal variables the isotone projection is computed by PAVA [@barlow1972statistical], a direct translation of the Fortran AMALGM routine (Algorithm AS 149, @kruskal1964nonmetric). Given a sequence $y_1, \ldots, y_k$ of group means, PAVA maintains a list of *blocks* and repeatedly merges adjacent violating pairs (pairs where $y_i > y_{i+1}$) by replacing them with their weighted mean, until the sequence is non-decreasing.

## Initialization

A key design decision is the initialization of $\mathbf{X}$ before the first ALS iteration. R's implementation uses a seeded random normal matrix, but R and NumPy produce different streams for the same seed. `pyGifi` instead uses a deterministic SVD-based initialization: the top-$d$ left singular vectors of the mean-centred, horizontally stacked basis matrix $[\mathbf{G}_1 | \cdots | \mathbf{G}_J]$. This is the exact closed-form solution when all variables are metric (standard PCA), and the best linear approximation for mixed-type data. For researchers who require exact numerical parity with R, a compiled C extension (`pygifi_rng`) ports R's Mersenne-Twister [@matsumoto1998mersenne] and AS241 normal quantile inversion routine exactly.

# Features

`pyGifi` implements the following Gifi models:

| Class | Method | Description |
|---|---|---|
| `Homals` | Homogeneity Analysis | MCA-equivalent for nominal data; finds object scores and category centroids that maximise between-category discrimination |
| `Princals` | Optimal Scaling PCA | Generalises PCA to nominal, ordinal, and metric variables simultaneously |
| `Morals` | Monotone Regression | Nonlinear multiple regression where both predictors and response are optimally scaled |
| `Corals` | Correlational Analysis | Maximises correlation between two sets of optimally scaled variables |
| `Canals` | Canonical Correlation | Canonical correlation analysis with optimal scaling |
| `Criminals` | Discriminant Analysis | Nonlinear discriminant analysis |
| `Overals` | Multiset Analysis | Generalises Homals/Corals to multiple variable sets |
| `Primals` | Primal Regression | Regression with metric response and optimally scaled predictors |
| `Addals` | Additive Analysis | Additive models with optimal scaling |
| `GifiIterativeImputer` | Missing Data | Iterative imputation within the Gifi ALS framework |

Additional utilities include:

- `knots_gifi()` — B-spline knot placement matching R's `knotsGifi()` exactly (quantile, regular, dense, or empty types).
- `cv_morals()` — $k$-fold cross-validation for Morals with optimal scaling.
- `pygifi.plot()` — unified plot dispatcher for object score plots, biplots, transformation plots, and loading plots.
- `get_dataset()` — 12 built-in classic datasets used in the Gifi literature.
- `categorical_encode()` / `categorical_decode()` — categorical coding utilities.

# Validation

Correctness is verified through a three-phase automated validation suite (`compare_test.py`) that runs both `pyGifi` and R's `Gifi` on the same datasets and compares outputs:

1. **Phase 1 — Numerical accuracy**: Category-by-category diff of model outputs (object scores, component loadings, category quantifications, eigenvalues) with tolerance `1e-3` to account for BLAS and platform differences.
2. **Phase 2 — Distribution comparison**: Comparison of empirical skewness, kurtosis, and overlaid histogram visualizations of transformed variable distributions.
3. **Phase 3 — Structural PCA comparison**: Macro-structural comparison of eigenvalue spectra, object score scatter plots, and loading scatter plots.

For users with the `pygifi_rng` C extension compiled, exact parity to `1e-6` is achieved by reproducing R's random initialization exactly. The unit test suite (`pytest tests/`) contains over 30 test modules covering individual primitives (PAVA, Dykstra, B-splines, Modified Gram-Schmidt, indicator matrices) through to full end-to-end model outputs validated against stored R fixtures.

# Example Usage

```python
import pygifi

# Homogeneity Analysis on the built-in Hartigan dataset
df = pygifi.get_dataset('hartigan')
model = pygifi.Homals(ndim=2, levels='nominal')
model.fit(df)
print(model)
pygifi.plot(model, plot_type='objplot')

# Optimal scaling PCA on mixed-type data
df = pygifi.get_dataset('galo')
model = pygifi.Princals(
    ndim=2,
    levels=['nominal', 'ordinal', 'nominal', 'ordinal']
)
model.fit(df)
pygifi.plot(model, plot_type='biplot')

# Monotone regression
df = pygifi.get_dataset('neumann')
X, y = df.iloc[:, :-1], df.iloc[:, -1]
model = pygifi.Morals(xdegrees=2, ydegrees=2, xordinal=True, yordinal=True)
model.fit(X, y)
```

# Acknowledgements

We thank Patrick Mair, Jan de Leeuw, and Patrick Groenen for the original R `Gifi` package and for making the source code openly available under the GPL-3.0 licence, which made this port possible. `pyGifi` is licensed under GPL-3.0-or-later, consistent with the original.

# References
