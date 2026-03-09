"""
pygifi.morals — Monotone Regression Analysis (Morals).

Python port of Gifi/R/morals.R.
R original: morals() by Mair, De Leeuw, Groenen. GPL-3.0.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin

from ._coding import make_numeric
from ._utilities import reshape
from ._splines import knots_gifi
from ._structures import make_gifi
from ._engine import gifi_engine
from ._linalg import ls_rc


class Morals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Monotone Regression Analysis via Optimal Scaling.

    Python port of R's morals(). Regresses y on X with optimal monotone
    transformation of both predictor and response variables.

    Parameters
    ----------
    xknots : list, default=None
    yknots : list, default=None
    xdegrees : int or list, default=2
    ydegrees : int, default=2
    xordinal : bool or list, default=True
    yordinal : bool, default=True
    xties : str or list, default="s"
    yties : str, default="s"
    xmissing : str or list, default="m"
    ymissing : str, default="m"
    xactive : bool or list, default=True
    xcopies : int or list, default=1
    itmax : int, default=1000
        Maximum number of ALS iterations.
    eps : float, default=1e-6
        Convergence criterion.
    verbose : bool, default=False
        Print iteration info.
    """

    def __init__(
        self,
        xknots: Optional[List[np.ndarray]] = None,
        yknots: Optional[List[np.ndarray]] = None,
        xdegrees: Union[int, List[int]] = 2,
        ydegrees: int = 2,
        xordinal: Union[bool, List[bool]] = True,
        yordinal: bool = True,
        xties: Union[str, List[str]] = "s",
        yties: str = "s",
        xmissing: Union[str, List[str]] = "m",
        ymissing: str = "m",
        xactive: Union[bool, List[bool]] = True,
        xcopies: Union[int, List[int]] = 1,
        itmax: int = 1000,
        eps: float = 1e-6,
        verbose: bool = False,
        init_x: Optional[np.ndarray] = None,
    ) -> None:
        self.xknots = xknots
        self.yknots = yknots
        self.xdegrees = xdegrees
        self.ydegrees = ydegrees
        self.xordinal = xordinal
        self.yordinal = yordinal
        self.xties = xties
        self.yties = yties
        self.xmissing = xmissing
        self.ymissing = ymissing
        self.xactive = xactive
        self.xcopies = xcopies
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, pd.DataFrame, np.ndarray], sample_weight: Any = None) -> 'Morals':
        """Fit Morals model to predictors X and response y."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y)
            
        npred, nobs = X.shape[1], X.shape[0]
        
        # Coerce to numeric (handles categories/factors like R)
        X_num = make_numeric(X)
        y_num = make_numeric(y).ravel()  # type: ignore
        
        xnames = list(X.columns)

        # Broadcast per-predictor params
        xdegrees_v = reshape(self.xdegrees, npred)  # type: ignore
        xordinal_v = reshape(self.xordinal, npred)  # type: ignore
        xties_v = reshape(self.xties, npred)  # type: ignore
        xmissing_v = reshape(self.xmissing, npred)  # type: ignore
        xactive_v = reshape(self.xactive, npred)  # type: ignore
        xcopies_v = reshape(self.xcopies, npred)  # type: ignore

        # Default knots
        if self.xknots is None:
            xknots = [
                knots_gifi(X.iloc[:, [i]], type="Q", n=None)[0]  # type: ignore
                for i in range(npred)
            ]
        else:
            xknots = self.xknots
            
        if self.yknots is None:
            yknots = [knots_gifi(pd.DataFrame(y), type="Q", n=None)[0]]  # type: ignore
        else:
            yknots = self.yknots

        # Combine X+y into one data matrix
        data = np.column_stack([X_num, y_num])
        
        # All X share set 0, Y is set 1
        sets = [0] * npred + [1]

        gifi = make_gifi(  # type: ignore
            data=pd.DataFrame(data),
            knots=xknots + yknots,
            degrees=xdegrees_v + [self.ydegrees],
            ordinal=xordinal_v + [self.yordinal],
            ties=xties_v + [self.yties],
            copies=xcopies_v + [1],
            missing=xmissing_v + [self.ymissing],
            active=xactive_v + [True],
            names=xnames + ["Y"],
            sets=sets,
        )

        h = gifi_engine(  # type: ignore
            gifi, ndim=1, itmax=self.itmax, eps=self.eps,
            verbose=self.verbose, init_x=self.init_x
        )

        # Collect transforms
        xhat = np.hstack([h["xGifi"][0][j]["transform"] for j in range(npred)])
        yhat = h["xGifi"][1][0]["transform"][:, 0]

        # OLS: xhat -> yhat
        beta = ls_rc(xhat, yhat)["solution"].flatten()  # type: ignore
        ypred = xhat @ beta
        yres = yhat - ypred
        smc = float(yhat @ ypred)

        rhat = np.corrcoef(np.column_stack([xhat, yhat]).T)

        self.result_ = {
            "rhat": rhat,
            "objectscores": h["x"],
            "xhat": xhat,
            "yhat": yhat,
            "beta": beta,
            "ypred": ypred,
            "yres": yres,
            "smc": smc,
            "f": h["f"],
            "ntel": h["ntel"],
            "xknots": xknots,
            "yknots": yknots,
            "xordinal": xordinal_v,
            "yordinal": self.yordinal,
        }
        self.n_obs_ = nobs
        self.n_pred_ = npred
        self.n_iter_ = h["ntel"]
        self.X_ = X
        self.y_ = y
        self.converged_ = h["ntel"] < self.itmax
        self.is_fitted_ = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Returns the object scores."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "is_fitted_")
        return self.result_["objectscores"]  # type: ignore

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            return (f"Morals()\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Squared Multiple Correlation (SMC): {self.result_['smc']:.6f}")
        return "Morals()"
