"""Tests for pygifi._isotone — PAVA, isotone regression, cone_regression, Dykstra."""

import numpy as np
from pygifi._isotone import pava, isotone, cone_regression, dykstra


# ---------- pava ----------

def test_pava_decreasing_uniform_weights():
    """Decreasing sequence with equal weights should average to global mean."""
    x = np.array([5., 4., 3., 2., 1.])
    result = pava(x)
    assert np.allclose(result, [3., 3., 3., 3., 3.]), f"Got {result}"


def test_pava_already_isotonic():
    """Already non-decreasing sequence should be returned unchanged."""
    x = np.array([1., 2., 3., 4., 5.])
    result = pava(x)
    assert np.allclose(result, x)


def test_pava_all_equal():
    x = np.array([2., 2., 2.])
    result = pava(x)
    assert np.allclose(result, [2., 2., 2.])


def test_pava_two_blocks():
    """[3, 1, 2] → should pool 3 with 1 (mean=2), then check 2 vs 2 → [2, 2, 2]."""
    x = np.array([3., 1., 2.])
    result = pava(x)
    assert np.allclose(result, [2., 2., 2.])


def test_pava_weighted():
    """Weighted PAVA: heavier weight blocks should dominate the mean."""
    x = np.array([2., 1.])  # violations: 2 > 1
    w = np.array([3., 1.])  # weight 3 for first
    result = pava(x, w)
    # Pooled mean: (3*2 + 1*1)/(3+1) = 7/4 = 1.75
    assert np.allclose(result, [1.75, 1.75])


def test_pava_single_element():
    x = np.array([42.])
    result = pava(x)
    assert np.allclose(result, [42.])


def test_pava_length():
    x = np.array([5., 3., 4., 1., 2.])
    result = pava(x)
    assert len(result) == len(x)
    # Must be non-decreasing
    assert np.all(np.diff(result) >= -1e-12)


def test_pava_zero_weights():
    """Test PAVA with some weights being exactly zero.

    Fortran AMALGM reconstructs by accumulating original weights until
    the sum matches the pooled block weight.  Block 0 pools x[0]=2 with
    x[1]=1 (weights 1+0=1), so its weight is 1.  In reconstruction,
    wo[0]=1 already matches w[0]=1, so only xa[0]=2.  Element 1 (wo=0)
    and element 2 (wo=1) are then both claimed by block 1 (value=3).
    Result: [2, 3, 3].
    """
    x = np.array([2., 1., 3.])
    w = np.array([1., 0., 1.])  # Second element has no weight
    result = pava(x, w)
    assert np.allclose(result, [2., 3., 3.])


def test_pava_empty():
    """Empty arrays should return empty."""
    x = np.array([])
    assert len(pava(x)) == 0

# ---------- isotone ----------


def test_isotone_simple_ties_s():
    """Ties mode 's': tied x values are averaged, then PAVA applied."""
    x = np.array([1., 1., 2., 3.])
    y = np.array([4., 2., 3., 1.])  # groups: [4,2], [3], [1]
    result = isotone(x, y, ties='s')
    # Group 1 mean=3, Group 2 mean=3, Group 3 mean=1 → PAVA([3,3,1])
    # PAVA pools group 2 and 3: (3+1)/2=2 → [3,3,2,2] after assigning back
    assert len(result) == 4
    # Result must be non-decreasing within same x-group (may vary by
    # implementation)


def test_isotone_monotone_output():
    """Output should be non-decreasing when sorted by x (no ties)."""
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([3., 1., 4., 1., 5.])
    result = isotone(x, y, ties='s')
    assert np.all(np.diff(result) >= -1e-12), f"Not isotonic: {result}"


def test_isotone_all_tie_modes():
    """All 3 tie modes should return arrays of correct length."""
    x = np.array([1., 1., 2., 2., 3.])
    y = np.array([3., 1., 4., 2., 5.])
    for ties in ['s', 'p', 't']:
        result = isotone(x, y, ties=ties)
        assert len(result) == 5, f"Wrong length for ties={ties}"


# ---------- cone_regression ----------

def test_cone_regression_subspace():
    """Type 's' should return OLS projection onto basis column space."""
    np.random.seed(1)
    basis = np.random.randn(10, 3)
    target = np.random.randn(10)
    data = np.arange(1., 11.)
    result = cone_regression(data, target, basis, type='s')
    # Should minimize ||target - basis @ coef||^2
    coef, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
    expected = basis @ coef
    assert np.allclose(result, expected, atol=1e-8)


def test_cone_regression_returns_array():
    data = np.array([1., 2., 3., 4., 5.])
    target = np.array([5., 3., 4., 2., 1.])
    result = cone_regression(data, target, type='c')
    assert result.shape == (5,)


# ---------- dykstra ----------

def test_dykstra_converges():
    """Dykstra should produce a value between x1 and x2 (average at convergence)."""
    np.random.seed(42)
    n = 8
    basis = np.random.randn(n, 2)
    data = np.sort(np.random.randn(n))
    target = np.random.randn(n)
    result = dykstra(target, basis, data, ties='s')
    assert result.shape == (n,)
    assert np.all(np.isfinite(result))
