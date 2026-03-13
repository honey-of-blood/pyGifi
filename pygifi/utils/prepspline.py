# mypy: ignore-errors
"""
pygifi._prepspline — Level-to-spline parameter mapping.

Python port of Gifi/R/prepspline.R (Mair, De Leeuw, Groenen. GPL-3.0).

Functions
---------
level_to_spline : R level_to_spline — convert level strings to knots + ordinal flags
"""

from pygifi.utils.splines import knots_gifi
import pandas as pd


def level_to_spline(levels, data):
    """
    Convert measurement level strings to knot lists and ordinal flags.

    Python port of R's level_to_spline(levels, data).

    R prepspline.R mapping:
        'nominal'  → knotsGifi(x, "D") — unique-value knots, ordinal=False, degree=-1
        'ordinal'  → knotsGifi(x, "Q") — quantile knots,      ordinal=True,  degree=-1
        'metric'   → knotsGifi(x, "E") — no interior knots,   ordinal=True,  degree=1

    Parameters
    ----------
    levels : list of str — one per column: 'nominal', 'ordinal', or 'metric'
    data   : np.ndarray (n, nvars) — numeric data matrix (already make_numeric'd)

    Returns
    -------
    dict with:
        'knotList' : list of np.ndarray, one per column
        'ordvec'   : list of bool, one per column
        'degvec'   : list of int, one per column (-1=categorical, 1=linear, etc.)
    """
    VALID = ('nominal', 'ordinal', 'metric')
    levels = [lv.lower() for lv in levels]
    for lv in levels:
        if lv not in VALID:
            raise ValueError(f"level must be one of {VALID}, got '{lv}'")

    if data.ndim == 1:
        data = data[:, None]

    knot_list = []
    ord_vec = []
    deg_vec = []

    for i, lv in enumerate(levels):
        col = data[:, i]
        df_col = pd.DataFrame(col)

        if lv == 'nominal':
            # R: knotsGifi(x, "D") — knots at unique data values (drop endpoints)
            kl = knots_gifi(df_col, type='D')
            knot_list.append(kl[0])
            ord_vec.append(False)
            deg_vec.append(-1)   # categorical indicator basis

        elif lv == 'ordinal':
            # R: knotsGifi(x, "Q") — quantile knots; ordinal=True for PAVA
            kl = knots_gifi(df_col, type='Q')
            knot_list.append(kl[0])
            ord_vec.append(True)
            deg_vec.append(-1)   # categorical ordinal by default

        elif lv == 'metric':
            # R: knotsGifi(x, "E") — no interior knots; linear polynomial
            kl = knots_gifi(df_col, type='E')
            knot_list.append(kl[0])
            ord_vec.append(True)
            deg_vec.append(1)    # linear (degree=1) continuous transform

    return {
        'knotList': knot_list,
        'ordvec': ord_vec,
        'degvec': deg_vec,
    }
