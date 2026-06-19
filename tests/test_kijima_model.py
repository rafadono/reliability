import numpy as np
import pytest
from src.reliability_analysis.analysis.kijima_model import (
    KijimaModelI,
    KijimaModelII,
)


def test_kijima_ii_happy_path():
    """Verify Kijima II recurrence: v_i = a_i * (v_{i-1} + x_i)."""
    x = np.array([10.0, 20.0, 30.0])
    delta = np.array([1, 0, 1], dtype=float)
    ar, ap = 0.5, 0.1

    # i=0: a=ar=0.5, v0 = 0.5*(0+10) = 5.0
    # i=1: a=ap=0.1, v1 = 0.1*(5+20) = 2.5
    # i=2: a=ar=0.5, v2 = 0.5*(2.5+30) = 16.25
    expected = np.array([5.0, 2.5, 16.25])
    model = KijimaModelII(beta=1.5, eta=500.0, ar=ar, ap=ap)
    result = model.virtual_age(x, delta)
    np.testing.assert_array_almost_equal(result, expected)


def test_kijima_ii_as_bad_as_old():
    """ar=ap=1 -> virtual age equals cumulative sum of intervals."""
    x = np.array([10.0, 20.0, 30.0])
    delta = np.array([1, 0, 1], dtype=float)
    expected = np.cumsum(x)
    model = KijimaModelII(beta=1.5, eta=500.0, ar=1.0, ap=1.0)
    result = model.virtual_age(x, delta)
    np.testing.assert_array_almost_equal(result, expected)


def test_kijima_ii_as_good_as_new():
    """ar=ap=0 -> virtual age is zero after every event."""
    x = np.array([10.0, 20.0, 30.0])
    delta = np.array([1, 0, 1], dtype=float)
    model = KijimaModelII(beta=1.5, eta=500.0, ar=0.0, ap=0.0)
    result = model.virtual_age(x, delta)
    np.testing.assert_array_almost_equal(result, np.zeros(3))


def test_kijima_ii_single_event():
    """Single event: v0 = ar * x0."""
    x = np.array([10.0])
    delta = np.array([1], dtype=float)
    ar = 0.5
    model = KijimaModelII(beta=1.5, eta=500.0, ar=ar, ap=0.1)
    result = model.virtual_age(x, delta)
    np.testing.assert_array_almost_equal(result, [ar * 10.0])
