"""
pygifi.homals — Multiple Correspondence Analysis (Homals).

Python port of Gifi/R/homals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Homals finds a low-dimensional representation of categorical (or mixed) data
by finding object scores and category quantifications that best fit each other
in a least-squares sense, using Alternating Least Squares with optimal scaling.

Compatible with scikit-learn: BaseEstimator, TransformerMixin, fit/transform API.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pygifi._coding import make_numeric
from pygifi._utilities import reshape, cor_list
from pygifi._prepspline import level_to_spline
from pygifi._structures import make_gifi
from pygifi._engine import gifi_engine


class Homals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Multiple Correspondence Analysis via ALS optimal scaling.

    Python port of R's homals() function (Gifi package).

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions (components).
    levels : str or list of str, default='nominal'
        Measurement level per variable. One of:
        'nominal'  — categorical, no order (indicator basis, subspace transform)
        'ordinal'  — categorical with order (indicator basis, isotone transform)
        'metric'   — continuous (polynomial basis, isotone transform)
        A single string is broadcast to all variables.
    ties : str, default='s'
        Tie-handling for isotone regression: 's' (secondary), 'p' (primary),
        't' (tertiary).
    missing : str, default='s'
        Missing value treatment: 'm' (individual), 'a' (average), 's' (shared).
    normobj_z : bool, default=True
        If True, scale objectscores by sqrt(nobs), matching R's default.
    active : bool or list of bool, default=True
        Which variables participate actively in ALS.
    itmax : int, default=1000
        Maximum ALS iterations.
    eps : float, default=1e-6
        Convergence threshold on loss change.
    verbose : bool, default=False
        Print iteration details.

    Attributes
    ----------
    result_ : dict
        Dictionary with all fitted quantities:
        - objectscores    : (nobs, ndim) — object coordinates
        - quantifications : list of (nbasis, ndim) arrays — category coordinates
        - transform       : list of (nobs, copies) arrays — optimal transforms
        - weights         : list of (copies, ndim) arrays — regression weights
        - loadings        : list of (ndim, copies) arrays — x.T @ transform
        - rhat            : (sum_copies, sum_copies) correlation matrix
        - evals           : eigenvalues of rhat
        - lambda_         : (ndim, ndim) average discriminant matrix
        - f               : final loss value
        - ntel            : number of iterations
    n_obs_ : int
    n_vars_ : int
    n_iter_ : int
    converged_ : bool
    """

    def __init__(
        self, 
        ndim: int = 2, 
        levels: Union[str, List[str]] = 'nominal', 
        ordinal: Optional[Union[bool, List[bool]]] = None, 
        knots: Optional[List[np.ndarray]] = None,
        ties: Union[str, List[str]] = 's', 
        degrees: Union[int, List[int]] = -1, 
        missing: Union[str, List[str]] = 's', 
        normobj_z: bool = True,
        active: Union[bool, List[bool]] = True, 
        itmax: int = 1000, 
        eps: float = 1e-6, 
        verbose: bool = False, 
        init_x: Optional[np.ndarray] = None
    ) -> None:
        self.ndim = ndim
        self.levels = levels
        self.ordinal = ordinal
        self.knots = knots
        self.ties = ties
        self.degrees = degrees
        self.missing = missing
        self.normobj_z = normobj_z
        self.active = active
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> 'Homals':
        """
        Fit Homals on X.

        Parameters
        ----------
        X : pd.DataFrame or array-like (nobs, nvars)
        y : ignored (scikit-learn API compatibility)

        Returns
        -------
        self
        """
        # --- Coerce input ---
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        names = list(X.columns)
        list(X.index)
        data_orig = X.copy()

        data = make_numeric(X)           # (nobs, nvars) numeric array
        nobs, nvars = data.shape

        # --- Validate + broadcast scalar params ---
        valid_ties = ('s', 'p', 't')
        valid_missing = ('m', 's', 'a')
        if self.ties not in valid_ties:
            raise ValueError(f"ties must be one of {valid_ties}, got '{self.ties}'")
        if self.missing not in valid_missing:
            raise ValueError(f"missing must be one of {valid_missing}, got '{self.missing}'")

        ties_v = reshape(self.ties, nvars)  # type: ignore
        missing_v = reshape(self.missing, nvars)  # type: ignore
        active_v = reshape(self.active, nvars)  # type: ignore
        levels_v = reshape(self.levels, nvars)  # type: ignore

        # --- Prepare spline/knot params from levels ---
        if self.knots is None:
            levelprep = level_to_spline(levels_v, data)  # type: ignore
            ordinal_v = levelprep['ordvec']
            knots_v = levelprep['knotList']
            degrees_v = levelprep['degvec']
        else:
            ordinal_v = reshape(self.ordinal if self.ordinal is not None else True, nvars)  # type: ignore
            knots_v = self.knots
            degrees_v = reshape(self.degrees, nvars)  # type: ignore

        copies_v = [self.ndim] * nvars   # each variable gets ndim copies

        # --- Build Gifi structure ---
        gifi = make_gifi(  # type: ignore
            data=data,
            knots=knots_v,
            degrees=degrees_v,
            ordinal=ordinal_v,
            ties=ties_v,
            copies=copies_v,
            missing=missing_v,
            active=active_v,
            names=names,
            sets=list(range(nvars)),  # each variable is its own set (homals convention)
        )

        # --- Run ALS engine ---
        h = gifi_engine(gifi, ndim=self.ndim, itmax=self.itmax,  # type: ignore
                        eps=self.eps, verbose=self.verbose,
                        init_x=self.init_x)

        # --- Assemble result dict (mirrors homals.R output) ---
        xGifi = h['xGifi']

        transform = []
        weights = []
        scores = []
        quantifications = []
        loadings = []
        dsum = np.zeros((self.ndim, self.ndim))
        nact = 0

        dmeasures = []
        for j in range(nvars):
            jgifi = xGifi[j][0]   # set j, variable 0 (each var in own set)
            tr_j = jgifi['transform']
            w_j = jgifi['weights']
            sc_j = jgifi['scores']
            q_j = jgifi['quantifications']

            transform.append(tr_j)
            weights.append(w_j)
            scores.append(sc_j)
            quantifications.append(q_j)

            cy = sc_j.T @ sc_j                           # (ndim, ndim)
            dmeasures.append(cy)
            if gifi[j][0]['active']:
                dsum += cy
                nact += 1

        # Correlation matrix across all transforms
        rhat = cor_list(transform)  # type: ignore
        evals_raw = np.linalg.eigh(rhat)[0]
        evals = evals_raw[::-1]

        # Object scores (with optional sqrt(nobs) scaling)
        objectscores = h['x'].copy()
        if self.normobj_z:
            objectscores = objectscores * np.sqrt(nobs)

        # Bug 3: R Object Score Normalization (variance = 1 per dimension)
        for d in range(self.ndim):
            std = np.std(objectscores[:, d])
            if std > 1e-10:
                objectscores[:, d] /= std
                # also rescale quantifications consistently
                for q in quantifications:
                    q[:, d] /= std

        lambda_ = dsum / nvars                           # average discriminant

        # Score matrix: first dim of each variable's scores (matches R scoremat)
        scoremat = np.column_stack([sc[:, 0] for sc in scores])

        # Bug 2: Proper per-variable dictionary for quantifications and loadings
        self.quantifications = {name: q for name, q in zip(names, quantifications)}
        self.loadings = {name: q.T for name, q in zip(names, quantifications)}
        # Also expose via trailing underscore for scikit-learn convention
        self.quantifications_ = self.quantifications
        self.loadings_ = self.loadings

        self.result_ = {
            'transform': transform,
            'rhat': rhat,
            'evals': evals,
            'objectscores': objectscores,
            'scoremat': scoremat,
            'quantifications': self.quantifications,
            'dmeasures': dmeasures,
            'weights': weights,
            'loadings': self.loadings,
            'lambda_': lambda_,
            'f': h['f'],
            'ntel': h['ntel'],
            'data': data_orig,
            'datanum': data,
            'ndim': self.ndim,
            'knots': knots_v,
            'degrees': degrees_v,
            'ordinal': ordinal_v,
        }

        self.n_obs_ = nobs
        self.n_vars_ = nvars
        self.n_iter_ = h['ntel']
        self.converged_ = (h['ntel'] < self.itmax)
        self.is_fitted_ = True

        return self

    def transform(self, X: Any, y: Any = None) -> np.ndarray:
        """
        Return object scores for the fitted data.

        Note: Homals does not support out-of-sample projection (matches R).
        The input X is accepted for sklearn API compatibility but must be
        the same data used in fit().

        Parameters
        ----------
        X : ignored (out-of-sample not supported)

        Returns
        -------
        np.ndarray of shape (nobs, ndim) — object scores
        """
        check_is_fitted(self, 'is_fitted_')
        return self.result_['objectscores']  # type: ignore

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Homals(ndim={self.ndim})\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Homals(ndim={self.ndim})"
        
    def summary(self) -> None:
        """Print extended summary matching R's summary.homals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nCategory Quantifications:")
        for name, q in self.result_['quantifications'].items():
            print(f"-- {name} --")
            print(pd.DataFrame(q, columns=[f"D{d+1}" for d in range(self.ndim)]))
            print()
