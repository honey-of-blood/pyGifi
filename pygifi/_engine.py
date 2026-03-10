# mypy: ignore-errors
"""
pygifi._engine — Core ALS engine and transformation routing.

Python port of Gifi/R/gifiEngine.R + Gifi/R/gifiTransform.R
(Mair, De Leeuw, Groenen. GPL-3.0).

Functions
---------
gifi_transform : R gifiTransform — routes to cone_regression by (degree, ordinal)
gifi_engine    : R gifiEngine   — full ALS loop for all Gifi methods
"""

import numpy as np
from pygifi._linalg import gs_rc, ls_rc
from pygifi._utilities import center, normalize
from pygifi._isotone import cone_regression
from pygifi._structures import make_x_gifi


def gifi_transform(
        data,
        target,
        basis,
        copies,
        degree,
        ordinal,
        ties,
        missing):
    """
    Apply optimal scaling transformation to one variable.

    Python port of R's gifiTransform(data, target, basis, copies, degree, ordinal, ties, missing).

    Routes to cone_regression based on measurement level:
    - degree == -1, not ordinal → subspace ('s'): indicator basis OLS
    - degree == -1, ordinal     → categorical ('c'): monotone on categories
    - degree >= 0, not ordinal  → subspace ('s'): spline/polynomial OLS
    - degree >= 0, ordinal      → Dykstra ('i'):  spline+isotone intersection
    Extra copies (l >= 1) always use subspace ('s').

    Parameters
    ----------
    data    : np.ndarray (n,)
    target  : np.ndarray (n, copies)
    basis   : np.ndarray (n, k)
    copies  : int
    degree  : int
    ordinal : bool
    ties    : str
    missing : str

    Returns
    -------
    np.ndarray (n, copies)
    """
    nobs = len(data)
    h = np.zeros((nobs, copies))

    # First copy: route by (degree, ordinal)
    t0 = target[:, 0]
    if degree == -1:
        if ordinal:
            h[:, 0] = cone_regression(data=data, target=t0, type='c',
                                      ties=ties, missing=missing)
        else:
            h[:, 0] = cone_regression(data=data, target=t0, basis=basis,
                                      type='s', missing=missing)
    else:  # degree >= 0
        if ordinal:
            h[:, 0] = cone_regression(data=data, target=t0, basis=basis,
                                      type='i', ties=ties, missing=missing)
        else:
            h[:, 0] = cone_regression(data=data, target=t0, basis=basis,
                                      type='s', ties=ties, missing=missing)

    # Extra copies: always subspace
    for copy_idx in range(1, copies):
        h[:,
          copy_idx] = cone_regression(data=data,
                                      target=target[:,
                                                    copy_idx],
                                      basis=basis,
                                      type='s',
                                      ties=ties,
                                      missing=missing)

    return h


def gifi_engine(gifi, ndim, itmax=1000, eps=1e-6, verbose=False, init_x=None):
    """
    Alternating Least Squares engine for all Gifi methods.

    Python port of R's gifiEngine(gifi, ndim, itmax, eps, verbose).

    Parameters
    ----------
    gifi    : list of lists — from make_gifi()
    ndim    : int — number of dimensions
    itmax   : int — maximum ALS iterations
    eps     : float — convergence threshold on loss change
    verbose : bool — print iteration info
    init_x  : np.ndarray (nobs, ndim), optional
        Custom initial random matrix. If provided, this is used directly
        instead of generating one with np.random.seed(123). To get exact
        parity with R, generate this matrix in R with:
            set.seed(123)
            x <- matrix(rnorm(nobs * ndim), nobs, ndim)
        and pass to Python. If None (default), uses NumPy's RNG with seed 123.

    Returns
    -------
    dict with keys: f, ntel, x, xGifi
    """
    # Get nobs from first variable of first set
    nobs = len(gifi[0][0]['data'])
    len(gifi)
    nvars = sum(len(s) for s in gifi)

    if nvars < 2:
        raise ValueError(f"gifi_engine requires at least two variables to perform multivariate analysis, but only {nvars} were provided.")

    # --- Initialization ---
    if init_x is not None:
        # User-provided initial matrix (e.g., from R for exact parity)
        x = np.asarray(init_x, dtype=float)
        if x.shape != (nobs, ndim):
            raise ValueError(
                f"init_x must have shape ({nobs}, {ndim}), got {x.shape}")
    else:
        # Default: NumPy RNG with seed 123 (mirrors R's set.seed(123))
        np.random.seed(123)
        x = np.random.randn(nobs, ndim)

    x = gs_rc(center(x))['q']

    xGifi = make_x_gifi(gifi, x)

    # Compute initial loss fold
    fold = 0.0
    asets = 0
    for i, gifi_set in enumerate(gifi):
        x_set = xGifi[i]
        ha = np.zeros((nobs, ndim))
        active_count = 0
        for j, gv in enumerate(gifi_set):
            if gv['active']:
                active_count += 1
                ha += x_set[j]['scores']
        if active_count > 0:
            asets += 1
            fold += np.sum((x - ha) ** 2)
    fold /= (asets * ndim)

    # --- ALS loop ---
    itel = 1
    while True:
        xz = np.zeros((nobs, ndim))
        fnew = 0.0
        fmid = 0.0

        for i, gifi_set in enumerate(gifi):
            x_set = xGifi[i]

            # Stack transforms of active variables: hh (nobs,
            # sum_active_copies)
            hh_parts = []
            active_count = 0
            for j, gv in enumerate(gifi_set):
                if gv['active']:
                    active_count += 1
                    hh_parts.append(x_set[j]['transform'])
            if active_count == 0:
                continue

            hh = np.hstack(hh_parts)               # (nobs, sum_copies)

            # OLS regression of hh onto x
            lf = ls_rc(hh, x)
            aa = lf['solution']                     # (sum_copies, ndim)
            rs = lf['residuals']                    # (nobs, ndim)

            # Step-size scaling: kappa = max eigenvalue of aa.T @ aa
            eigvals = np.linalg.eigh(aa.T @ aa)[0]
            kappa = float(eigvals.max())

            fmid += np.sum(rs ** 2)

            # ALS target update: target = hh + rs @ aa.T / kappa
            target = hh + rs @ aa.T / kappa         # (nobs, sum_copies)

            # Update each variable's transform
            hh_new_parts = []
            scopies = 0
            for j, gv in enumerate(gifi_set):
                jcopies = x_set[j]['transform'].shape[1]
                ja = aa[scopies:scopies + jcopies, :]        # (copies, ndim)
                # (nobs, copies)
                jtarget = target[:, scopies:scopies + jcopies]

                hj = gifi_transform(
                    data=gv['data'],
                    target=jtarget,
                    basis=gv['basis'],
                    copies=gv['copies'],
                    degree=gv['degree'],
                    ordinal=gv['ordinal'],
                    ties=gv['ties'],
                    missing=gv['missing'],
                )
                hj = gs_rc(normalize(center(hj)))[
                    'q']       # normalize transform

                # (nobs, ndim) scores
                sc = hj @ ja

                # Update mutable state
                xGifi[i][j]['transform'] = hj
                xGifi[i][j]['weights'] = ja
                xGifi[i][j]['scores'] = sc
                xGifi[i][j]['quantifications'] = ls_rc(gv['basis'], sc)[
                    'solution']

                if gv['active']:
                    hh_new_parts.append(hj)

                scopies += jcopies

            if len(hh_new_parts) > 0:
                hh_new = np.hstack(hh_new_parts)
                ha = hh_new @ aa
                xz += ha
                fnew += np.sum((x - ha) ** 2)

        fmid /= (asets * ndim)
        fnew /= (asets * ndim)

        if verbose:
            print(
                f"Iter {
                    itel:4d}  fold={
                    fold:.8f}  fmid={
                    fmid:.8f}  fnew={
                    fnew:.8f}")

        # Convergence check (matches R: itel > 1 required)
        if (itel == itmax or (fold - fnew) < eps) and itel > 1:
            break

        itel += 1
        fold = fnew
        x = gs_rc(center(xz))['q']

    return {
        'f': fnew,
        'ntel': itel,
        'x': x,
        'xGifi': xGifi,
    }
