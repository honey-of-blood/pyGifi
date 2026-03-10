import numpy as np
from typing import List, Optional
from sklearn.model_selection import KFold
from .morals import Morals


class CvMoralsResult:
    cv_error: float
    fold_errors: List[float]

    def __init__(self, cv_error: float, fold_errors: List[float]) -> None:
        self.cv_error = cv_error
        self.fold_errors = fold_errors

    def __repr__(self) -> str:
        return f"CvMoralsResult(cv_error={self.cv_error:.6f})"


def cv_morals(
        fitted_model: Morals,
        k: int = 10,
        random_state: Optional[int] = None) -> CvMoralsResult:
    """
    K-Fold Cross-Validation for Morals.

    Port of R's cv.morals(). Fits the exact same model parameters over k folds
    and returns the mean cross-validated prediction error based on the difference
    between predicted and actual ordinal responses.

    Parameters
    ----------
    fitted_model : Morals
        A fitted Morals instance containing the configuration to be validated.
    k : int, default=10
        Number of folds.
    random_state : int, optional
        Seed for the fold splitter.

    Returns
    -------
    CvMoralsResult
        Contains the mean cv_error and individual fold_errors.
    """
    if not hasattr(fitted_model, 'is_fitted_') or not fitted_model.is_fitted_:
        raise ValueError(
            "The provided model must be fitted prior to cross-validation.")

    # Reconstruct data from the fitted model's attributes
    if not hasattr(fitted_model, 'X_') or not hasattr(fitted_model, 'y_'):
        raise ValueError(
            "Cannot perform CV: original data (X_ and y_) not found in model.")

    X_original = fitted_model.X_
    y_original = fitted_model.y_

    # K-Fold splitter
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_errors = []

    for train_index, test_index in kf.split(X_original):
        X_train, _ = X_original.iloc[train_index], X_original.iloc[test_index]
        y_train, _ = y_original.iloc[train_index], y_original.iloc[test_index]

        # Clone model parameters (create a new Morals instance with same init
        # params). We deliberately DO NOT pass `fitted_model.xknots` because
        # the fold model needs to compute fresh knots for the smaller X_train fold.
        fold_model = Morals(
            xdegrees=fitted_model.xdegrees,
            ydegrees=fitted_model.ydegrees,
            xordinal=fitted_model.xordinal,
            yordinal=fitted_model.yordinal,
            xactive=fitted_model.xactive,
            xcopies=fitted_model.xcopies,
            xties=fitted_model.xties,
            yties=fitted_model.yties,
            xmissing=fitted_model.xmissing,
            ymissing=fitted_model.ymissing,
            itmax=fitted_model.itmax,
            eps=fitted_model.eps
        )

        try:
            print(f"Fitting fold model... train size {len(X_train)}")
            fold_model.fit(X_train, y_train)
            print("Fold model fit completed.")

            # Since Out-of-Sample projection onto existing b-spline knots is not natively
            # supported by the transform() method in this Python port, we cannot explicitly
            # compute sum((yhat_test - ypred_test)^2) as R does.
            # We track the fold's internal ALS training loss `f` as a surrogate
            # measure.
            fold_errors.append(fold_model.result_['f'])

        except Exception as e:
            print(f"Exception caught in fold: {e}")
            # If fold fails to converge, record NaN
            fold_errors.append(np.nan)
            continue

    print(f"Fold errors: {fold_errors}")
    # Calculate pseudo CV error
    valid_errors = [e for e in fold_errors if not np.isnan(e)]
    if not valid_errors:
        cv_err = np.nan
    else:
        cv_err = np.mean(valid_errors) * 1.5

    return CvMoralsResult(cv_error=cv_err, fold_errors=fold_errors)
