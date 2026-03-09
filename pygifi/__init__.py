# mypy: ignore-errors
"""
pygifi — Python port of the R Gifi library.

Multivariate Analysis with Optimal Scaling.

Original R package: Mair, De Leeuw, Groenen (GPL-3.0)
Python port: GPL-3.0-or-later
"""

from pygifi.homals import Homals
from pygifi.princals import Princals
from pygifi.morals import Morals
from pygifi._coding import make_numeric, encode, decode, categorical_encode, categorical_decode
from pygifi._splines import knots_gifi
from pygifi._engine import gifi_transform
from pygifi.datasets import get_dataset
from pygifi.cv import cv_morals
from pygifi._isotone import cone_regression

__version__ = "1.0.0"
from typing import List

__all__: List[str] = ["Homals", "Princals", "Morals", "make_numeric", "knots_gifi", "gifi_transform",
           "encode", "decode", "categorical_encode", "categorical_decode", 
           "get_dataset", "cv_morals", "cone_regression", "__version__"]
