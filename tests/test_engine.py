"""Tests for pygifi._engine (gifi_transform) and pygifi._prepspline (level_to_spline)."""

import numpy as np
import pytest
from pygifi._engine import gifi_transform
from pygifi._prepspline import level_to_spline
from pygifi._utilities import make_indicator


# ---------- gifi_transform ----------

def test_gifi_transform_subspace_shape():
    """degree=-1, ordinal=False → subspace projection, shape must be (n, copies)."""
    np.random.seed(0)
    data = np.array([1., 2., 1., 3., 2., 1., 3., 2.])
    basis = make_indicator(data)
    target = np.random.randn(8, 2)
    h = gifi_transform(data, target, basis, copies=2, degree=-1,
                       ordinal=False, ties='s', missing='s')
    assert h.shape == (8, 2)


def test_gifi_transform_ordinal_categorical():
    """degree=-1, ordinal=True → cone_regression type='c', result must be isotonic."""
    data = np.array([1., 2., 1., 3., 2., 1., 3., 2.])
    basis = make_indicator(data)
    np.random.seed(1)
    target = np.random.randn(8, 1)
    h = gifi_transform(data, target, basis, copies=1, degree=-1,
                       ordinal=True, ties='s', missing='s')
    assert h.shape == (8, 1)
    # Values for same data-level should be ~equal (isotone on groups)
    idx1 = np.where(data == 1.)[0]
    np.where(data == 2.)[0]
    # All obs with data==1 get same value (secondary tie rule)
    assert np.allclose(h[idx1, 0], h[idx1[0], 0], atol=1e-8)


def test_gifi_transform_single_copy():
    """copies=1 → shape (n, 1)."""
    data = np.array([1., 2., 1., 3., 2.])
    basis = make_indicator(data)
    target = np.random.randn(5, 1)
    h = gifi_transform(data, target, basis, copies=1, degree=-1,
                       ordinal=False, ties='s', missing='s')
    assert h.shape == (5, 1)


def test_gifi_transform_polynomial():
    """degree=0, ordinal=False → polynomial basis subspace."""
    data = np.linspace(1., 5., 10)
    from pygifi._splines import bspline_basis
    basis = bspline_basis(data, degree=0, innerknots=np.array([3.]))
    target = np.random.randn(10, 1)
    h = gifi_transform(data, target, basis, copies=1, degree=0,
                       ordinal=False, ties='s', missing='s')
    assert h.shape == (10, 1)


# ---------- level_to_spline ----------

def test_level_to_spline_nominal():
    data = np.column_stack([np.array([1., 2., 3., 2., 1.])])
    result = level_to_spline(['nominal'], data)
    assert result['ordvec'] == [False]
    assert isinstance(result['knotList'][0], np.ndarray)


def test_level_to_spline_ordinal():
    data = np.column_stack([np.array([1., 2., 3., 2., 1.])])
    result = level_to_spline(['ordinal'], data)
    assert result['ordvec'] == [True]


def test_level_to_spline_metric():
    data = np.column_stack([np.array([1., 2., 3., 4., 5.])])
    result = level_to_spline(['metric'], data)
    assert result['ordvec'] == [True]
    # metric → empty knots (type 'E')
    assert len(result['knotList'][0]) == 0


def test_level_to_spline_mixed():
    data = np.column_stack([
        np.array([1., 2., 3., 2., 1.]),
        np.array([1., 2., 3., 4., 5.]),
        np.array([1., 2., 1., 2., 1.]),
    ])
    result = level_to_spline(['nominal', 'metric', 'ordinal'], data)
    assert result['ordvec'] == [False, True, True]
    assert len(result['knotList']) == 3
    assert len(result['knotList'][1]) == 0   # metric → empty


def test_level_to_spline_invalid_raises():
    data = np.column_stack([np.array([1., 2., 3.])])
    with pytest.raises(ValueError, match="level must be"):
        level_to_spline(['fancy'], data)


def test_level_to_spline_returns_correct_keys():
    data = np.column_stack([np.array([1., 2., 1.])])
    result = level_to_spline(['nominal'], data)
    assert 'knotList' in result
    assert 'ordvec' in result
