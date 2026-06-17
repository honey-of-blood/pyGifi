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

Mixed-type data — datasets where some columns are categorical, others are ordinal, and others are continuous — is ubiquitous in the social sciences, healthcare, market research, and survey analysis. Classical dimensionality reduction tools such as PCA [@jolliffe2002principal] and factor analysis assume all variables are metric, meaning they impose arbitrary linear distances on categorical codes (e.g., treating "Red = 1, Blue = 2, Green = 3" as if Blue is numerically halfway between Red and Green). This assumption is statistically invalid and can severely distort downstream analysis.

The Gifi system resolves this by treating the numerical representation of each variable as an unknown to be *optimized*: optimal scaling finds the category quantifications that maximize the variance explained by the model while strictly respecting the variable's measurement level. The practical consequence is that analysts can apply classical PCA-like decompositions to any dataset, regardless of column types, without making invalid metric assumptions.

While R users have access to the mature `Gifi` [@mair2014gifi] and `homals` [@de2009gifi] packages, **no complete, actively-maintained Python implementation existed** prior to `pyGifi`. Existing Python alternatives are partial or limited:

- `prince` provides Multiple Correspondence Analysis and Factor Analysis of Mixed Data, but does not implement the full Gifi algorithm family, does not support ordinal constraints via PAVA, and has no B-spline basis support.
- `scikit-learn`'s `OrdinalEncoder` and `OneHotEncoder` encode categorical variables but do not optimize the encoding.
- `mca` implements a standalone MCA but is unmaintained and lacks the broader Gifi model family.

`pyGifi` fills this gap by providing a complete, validated Python port of the full Gifi algorithm family with a scikit-learn-compatible API, enabling researchers who work exclusively in Python to apply these methods without switching to R, and enabling reproducibility of published results that previously required proprietary or R-only software.

The library is intended for researchers in statistics, psychometrics, social sciences, and data science practitioners who work with mixed-type or predominantly categorical datasets.

# State of the Field

Optimal scaling and nonlinear multivariate analysis have a rich history rooted in psychometrics. The foundational Gifi system was formalized in @gifi1990nonlinear and brought to practical use in R through the `homals` package [@de2009gifi] and subsequently the unified `Gifi` package [@mair2014gifi]. Related work includes the ALSCAL algorithm [@young1981quantitative], which established Alternating Least Squares as the dominant computational framework for this class of problems, and the broader SMACOF (Scaling by Majorizing a Complicated Function) literature [@borg2005modern] which shares the majorization paradigm used in `pyGifi`'s optional `gifi_majorization` solver.

In Python, the ecosystem for categorical and mixed-type multivariate analysis remains underdeveloped relative to R. The `prince` library [@prince2023] provides MCA and FAMD (Factor Analysis of Mixed Data) but does not implement ordinal constraints or the complete Gifi model family. `FactorAnalyzer` supports classical factor analysis on metric data only. The `statsmodels` library [@seabold2010statsmodels] provides some multivariate methods but not optimal scaling. Commercial software such as IBM SPSS's CATPCA implements similar functionality but is proprietary and non-scriptable.

`pyGifi` is, to the authors' knowledge, the first open-source Python library to implement the complete Gifi family — including ordinal B-spline constraints, Dykstra's alternating projection, PAVA with three tie-handling modes, and the full suite of models from `Homals` through `Addals` — with validated numerical parity to a peer-reviewed R reference implementation.

# Software Design

`pyGifi` is organized into five layers, each with a single responsibility:

**1. Models layer** (`pygifi/models/`). Each Gifi algorithm is implemented as a scikit-learn `BaseEstimator` / `TransformerMixin` subclass with `fit()` and `transform()` methods. All models share the same constructor pattern: scalar hyperparameters are broadcast to per-variable lists, data is coerced and validated, the Gifi data structure is assembled, the shared ALS engine is called, and results are packed into a `result_` dictionary mirroring the corresponding R output field names.

**2. Core engine** (`pygifi/core/`). The `gifi_engine()` function in `engine.py` implements the main ALS loop shared by all models. It operates on a nested list-of-dicts data structure (`gifi` / `xGifi`) and routes each variable's transformation update through `gifi_transform()`, which dispatches to the appropriate cone projection based on `(degree, ordinal)`. The `structures.py` module contains factory functions that pre-compute basis matrices and pack variable metadata into the static `gifi` structure before the loop begins.

**3. Linear algebra primitives** (`pygifi/core/linalg.py`). All projections ultimately reduce to column-pivoted Modified Gram-Schmidt orthogonalization (`gs_rc`) and its derived least-squares solver (`ls_rc`). These are direct translations of R's `gsC.c` C source, preserving the exact pivot ordering and sign convention. Householder QR (as in `scipy.linalg.qr`) is intentionally avoided because it produces different Q columns and would break numerical parity.

**4. Transformation utilities** (`pygifi/utils/`). The `isotone.py` module implements PAVA (`pava`), isotone regression with three tie modes (`isotone`), and Dykstra's alternating projection (`dykstra`), all translated from R's Fortran and C sources. The `splines.py` module implements B-spline basis construction via the Cox–de Boor recursion [@deboor1978practical] using `scipy.interpolate.BSpline.design_matrix` with a boundary fix for the right endpoint. The `_cone.py` module is the single dispatch point for all cone projections (`project_cone`), providing six cone types: subspace (`'s'`), categorical isotone (`'c'`), Dykstra (`'i'`), monotone spline (`'m'`), linear-metric (`'l'`), and NNLS general cone (`'n'`).

**5. RNG C extension** (`pygifi/rng/`). To enable exact numerical parity with R's random initialization, a compiled C extension (`pygifi_rng`) ports R's Mersenne-Twister [@matsumoto1998mersenne] and the AS241 rational approximation for normal quantile inversion. This is optional: the default SVD-based initialization produces qualitatively equivalent results without requiring compilation.

The SVD-based deterministic initialization deserves special mention as a deliberate departure from R's behavior. R seeds a random normal matrix for initialization, but R's RNG and NumPy's produce different streams even for the same seed. `pyGifi` instead uses the top-$d$ left singular vectors of the horizontally stacked basis matrix as the starting point — the exact spectral solution for metric data and the best linear approximation for mixed-type data. This makes results fully reproducible without any seed argument, converges in fewer iterations, and avoids dependence on the C extension for standard use.

# Research Impact Statement

`pyGifi` enables several research workflows that were previously inaccessible to Python-only environments:

**Reproducibility of published research.** A large body of psychometric and social science literature reports results using R's `Gifi` or its predecessors. `pyGifi` allows these analyses to be reproduced and extended within Python pipelines, reducing the barrier to replication studies.

**Integration with modern Python ML stacks.** Because all models implement the scikit-learn `BaseEstimator` interface, `pyGifi` transformations can be used as preprocessing steps inside `sklearn.pipeline.Pipeline`, combined with classifiers, regressors, or clustering algorithms, and tuned with `GridSearchCV`. This makes optimal scaling accessible as a standard preprocessing layer rather than a standalone statistical tool.

**Mixed-type data preprocessing.** Many real-world datasets — clinical surveys, census records, customer profiles — contain heterogeneous columns that are routinely mis-handled by existing tools (one-hot encoding nominal columns without any structure, or treating ordinal codes as metric). `pyGifi` provides a principled alternative that finds the statistically optimal encoding for each column type simultaneously.

**Education and research in psychometrics.** The library ships with 12 classic datasets from the Gifi literature (Hartigan, Neumann, Roskam, Senate, Galo, etc.) and Jupyter notebook tutorials, making it suitable for teaching nonlinear multivariate analysis in Python-based courses.

# AI Usage Disclosure

During the preparation of this manuscript, Claude (Anthropic) was used to assist with drafting and structuring the paper text. All mathematical content, code, algorithmic descriptions, and references were reviewed and verified by the authors. The final text represents the authors' own work and scientific judgement.

# Acknowledgements

We thank Patrick Mair, Jan de Leeuw, and Patrick Groenen for the original R `Gifi` package and for making the source code openly available under the GPL-3.0 licence, which made this port possible. `pyGifi` is licensed under GPL-3.0-or-later, consistent with the original.

# References
