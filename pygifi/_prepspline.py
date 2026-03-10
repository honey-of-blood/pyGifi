# mypy: ignore-errors
"""
pygifi._prepspline — Level-to-spline parameter mapping.

Python port of Gifi/R/prepspline.R (Mair, De Leeuw, Groenen. GPL-3.0).

Functions
---------
level_to_spline : R level_to_spline — convert level strings to knots + ordinal flags
"""

from pygifi._splines import knots_gifi
import pandas as pd


def level_to_spline(levels, data):
    """
    Convert measurement level strings to knot lists and ordinal flags.

    Python port of R's level_to_spline(levels, data).

    Parameters
    ----------
    levels : list of str — one per column: 'nominal', 'ordinal', or 'metric'
    data   : np.ndarray (n, nvars) — numeric data matrix (already make_numeric'd)

    Returns
    -------
    dict with:
        'knotList' : list of np.ndarray, one per column
        'ordvec'   : list of bool, one per column
    """
    VALID = ('nominal', 'ordinal', 'metric')
    levels = [lv.lower() for lv in levels]
    for lv in levels:
        if lv not in VALID:
            raise ValueError(f"level must be one of {VALID}, got '{lv}'")

    data.shape[1] if data.ndim == 2 else 1
    if data.ndim == 1:
        data = data[:, None]

    knot_list = []
    ord_vec = []
    deg_vec = []

    for i, lv in enumerate(levels):
        col = data[:, i]
        df_col = pd.DataFrame(col)

        if lv == 'nominal':
            # R: knotsGifi(data[,i], "D") — D-type: unique values minus
            # endpoints
            kl = knots_gifi(df_col, type='D', n=3)
            knot_list.append(kl[0])
            ord_vec.append(False)
            deg_vec.append(-1)

        elif lv == 'ordinal':
            # R: knotsGifi(data[,i], "D") — same knots, but ordinal=True
            kl = knots_gifi(df_col, type='D', n=3)
            knot_list.append(kl[0])
            ord_vec.append(True)
            deg_vec.append(-1)  # Categorical ordinal by default

        elif lv == 'metric':
            # R: knotsGifi(data[,i], "E") — empty knots (polynomial/continuous)
            kl = knots_gifi(df_col, type='E', n=3)
            knot_list.append(kl[0])
            ord_vec.append(True)
            deg_vec.append(1)  # Linear polynomial by default

    return {
        'knotList': knot_list,
        'ordvec': ord_vec,
        'degvec': deg_vec,
    }
