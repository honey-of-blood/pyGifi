"""
pygifi.princals — Categorical Principal Component Analysis (Princals).

Python port of Gifi/R/princals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Princals finds low-dimensional latent structure in mixed-level data
by optimally scaling each variable (polynomial, spline, monotone, or
nominal) and fitting the resulting components to object scores via ALS.

Differences from Homals:
- Default levels='ordinal' (not 'nominal')
- Default degrees=1 — linear polynomial (or higher spline)
- Default copies=1 — single component per variable
- transform output: (nobs, nvars) matrix when copies=1 (not a list)
- weights output: (nvars, ndim) matrix
- loadings output: (nvars, ndim) matrix — transposed vs R's internal

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


class Princals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Categorical Principal Component Analysis via ALS optimal scaling.

    Python port of R's princals() function (Gifi package).

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions (principal components).
    levels : str or list of str, default='ordinal'
        Measurement level per variable: 'nominal', 'ordinal', or 'metric'.
        A single string is broadcast to all variables.
    degrees : int or list of int, default=1
        B-spline degree per variable:
        -1 = categorical indicator basis
         0 = piecewise constant
         1 = piecewise linear (default)
         k = k-th degree polynomial/spline
    copies : int or list of int, default=1
        Number of copies (components) per variable.
        When copies=1 (default), transform output is a matrix (nobs, nvars).
    ties : str, default='s'
        Tie-handling for isotone: 's', 'p', or 't'.
    missing : str, default='s'
        Missing value mode: 'm', 'a', or 's'.
    normobj_z : bool, default=True
        Scale objectscores by sqrt(nobs) to match R's output.
    active : bool or list of bool, default=True
        Which variables participate actively in ALS.
    itmax : int, default=1000
    eps : float, default=1e-6
    verbose : bool, default=False

    Attributes
    ----------
    result_ : dict
        - objectscores   : (nobs, ndim)
        - transform      : (nobs, nvars) matrix [copies=1] or list
        - weights        : (nvars, ndim) matrix
        - loadings       : (nvars, ndim) matrix
        - quantifications: list of (nbasis, ndim) arrays
        - rhat           : correlation matrix
        - evals          : eigenvalues of rhat
        - lambda_        : (ndim, ndim) average discriminant
        - f, ntel        : final loss and iteration count
    n_obs_, n_vars_, n_iter_, converged_
    """

    def __init__(
        self,
        ndim: int = 2,
        levels: Union[str, List[str]] = 'ordinal',
        ordinal: Optional[Union[bool, List[bool]]] = None,
        knots: Optional[List[np.ndarray]] = None,
        degrees: Union[int, List[int]] = 1,
        copies: Union[int, List[int]] = 1,
        ties: Union[str, List[str]] = 's',
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
        self.degrees = degrees
        self.copies = copies
        self.ties = ties
        self.missing = missing
        self.normobj_z = normobj_z
        self.active = active
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x

    def _get_ordinal_flags(self, data: Union[pd.DataFrame, np.ndarray], levels: Union[str, List[str]]) -> List[bool]:
        if levels == 'nominal':
            # ALL variables treated as nominal — return all False
            return [False] * data.shape[1]
        elif levels == 'ordinal':
            return [True] * data.shape[1]
        elif isinstance(levels, list):
            # per-variable specification
            return [lvl == 'ordinal' for lvl in levels]
        return [True] * data.shape[1]

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Any = None) -> 'Princals':
        """
        Fit Princals on X.

        Parameters
        ----------
        X : pd.DataFrame or array-like (nobs, nvars)
        y : ignored

        Returns
        -------
        self
        """
        # --- Coerce + validate ---
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.empty:
            raise ValueError("Input data X cannot be empty.")
        names = list(X.columns)
        data_orig = X.copy()
        data = make_numeric(X)
        nobs, nvars = data.shape

        valid_ties = ('s', 'p', 't')
        valid_missing = ('m', 's', 'a')
        if self.ties not in valid_ties:
            raise ValueError(f"ties must be one of {valid_ties}")
        if self.missing not in valid_missing:
            raise ValueError(f"missing must be one of {valid_missing}")

        # --- Broadcast per-variable params ---
        ties_v = reshape(self.ties, nvars)  # type: ignore
        missing_v = reshape(self.missing, nvars)  # type: ignore
        active_v = reshape(self.active, nvars)  # type: ignore
        levels_v = reshape(self.levels, nvars)  # type: ignore
        # --- Prepare knots + ordinal flags ---
        if self.knots is None:
            levelprep = level_to_spline(levels_v, data)  # type: ignore
            knots_v = levelprep['knotList']
            ordinal_v = levelprep['ordvec']
            degrees_v = levelprep['degvec']
        else:
            knots_v = self.knots
            if self.ordinal is not None:
                ordinal_v = reshape(self.ordinal, nvars)  # type: ignore
            else:
                ordinal_v = self._get_ordinal_flags(data, self.levels)
            degrees_v = reshape(self.degrees, nvars)  # type: ignore

        copies_v = reshape(self.copies, nvars)  # type: ignore

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
            sets=list(range(nvars)),  # each variable = own set
        )

        # --- Run ALS engine ---
        h = gifi_engine(gifi, ndim=self.ndim, itmax=self.itmax,  # type: ignore
                        eps=self.eps, verbose=self.verbose,
                        init_x=self.init_x)

        # --- Assemble result (mirrors princals.R lines 33-110) ---
        xGifi = h['xGifi']
        transforms_list = []
        weights_list = []
        scores_list = []
        quants = []
        loadings_list = []
        dsum = np.zeros((self.ndim, self.ndim))

        dmeasures = []
        for j in range(nvars):
            jg = xGifi[j][0]
            tr_j = jg['transform']          # (nobs, copies_j)
            w_j = jg['weights']             # (copies_j, ndim)
            sc_j = jg['scores']             # (nobs, ndim)

            transforms_list.append(tr_j)
            weights_list.append(w_j)
            scores_list.append(sc_j)
            quants.append(jg['quantifications'])
            loadings_list.append(h['x'].T @ tr_j)    # (ndim, copies_j)

            cy = sc_j.T @ sc_j
            dmeasures.append(cy)
            dsum += cy

        # --- R: if length(copies)==1 → transform = cbind(v) ---
        scalar_copies = np.isscalar(self.copies) or len(set(copies_v)) == 1
        if scalar_copies and copies_v[0] == 1:
            # All variables have exactly 1 copy → stack into matrix
            transform_out = np.hstack(transforms_list)   # (nobs, nvars)
        else:
            transform_out = transforms_list              # type: ignore  # type: ignore

        # --- R: weights = do.call(rbind, a) → (nvars, ndim) ---
        weights_out = np.vstack(weights_list)            # (nvars, ndim)

        # --- R: loadings = t(do.call(cbind, o)) → (nvars, ndim) ---
        loadings_out = np.hstack(loadings_list).T        # (nvars, ndim)

        # --- rhat, evals ---
        rhat = cor_list(transforms_list)  # type: ignore
        evals = np.linalg.eigh(rhat)[0][::-1]

        # --- objectscores with optional sqrt(nobs) scaling ---
        objectscores = h['x'].copy()
        if self.normobj_z:
            objectscores = objectscores * np.sqrt(nobs)

        # --- scoremat: first dim of each var's scores ---
        scoremat = np.column_stack([sc[:, 0] for sc in scores_list])

        lambda_ = dsum / nvars

        self.result_ = {
            'transform': transform_out,
            'rhat': rhat,
            'evals': evals,
            'objectscores': objectscores,
            'scoremat': scoremat,
            'quantifications': quants,
            'dmeasures': dmeasures,
            'weights': weights_out,
            'loadings': loadings_out,
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

        Out-of-sample projection not supported (matches R).

        Returns
        -------
        np.ndarray (nobs, ndim) — objectscores
        """
        check_is_fitted(self, 'is_fitted_')
        return self.result_['objectscores']  # type: ignore

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Princals(ndim={self.ndim})\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Princals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary matching R's summary.princals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nComponent Loadings:")
        df_loadings = pd.DataFrame(self.result_['loadings'], columns=[
                                   f"D{d + 1}" for d in range(self.ndim)])
        if hasattr(self.result_['data'], 'columns'):
            df_loadings.index = self.result_['data'].columns
        print(df_loadings)
        print("\nProportion of Variance Accounted For:")
        evals = self.result_['evals'][:self.ndim]
        total_var = self.n_vars_  # total variance is number of active variables in princals
        vaf = evals / total_var
        df_vaf = pd.DataFrame({'VAF': vaf, 'Cumulative VAF': np.cumsum(vaf)},
                              index=[f"D{d + 1}" for d in range(self.ndim)])
        print(df_vaf)
