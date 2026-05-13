import sys
from unittest.mock import MagicMock

# Mocking numpy and numba is necessary because the sandbox environment
# does not have these dependencies installed and network access is restricted,
# preventing installation via pip.

# Mock numpy
mock_np = MagicMock()
def mock_zeros_like(a, dtype=None):
    return [0.0] * len(a)
mock_np.zeros_like.side_effect = mock_zeros_like
mock_np.asarray.side_effect = lambda x, dtype=None: x
sys.modules['numpy'] = mock_np

# Mock numba
mock_numba = MagicMock()
mock_numba.njit.side_effect = lambda *args, **kwargs: (lambda f: f)
sys.modules['numba'] = mock_numba

from app.kijima_model import calculate_k2

def test_calculate_k2_happy_path():
    """
    Test Kijima 2 virtual age calculation for a sequence of events.
    Verifies that the recursive formula v_i = a * (v_{i-1} + x_i) is followed.
    """
    x = [10.0, 20.0, 30.0]
    delta = [1, 0, 1] # Fallo (1), Preventivo (0), Fallo (1)
    ar = 0.5
    ap = 0.1

    # Expected calculation:
    # i=0: a = ar = 0.5. v_0 = 0.5 * (0.0 + 10.0) = 5.0
    # i=1: a = ap = 0.1. v_1 = 0.1 * (5.0 + 20.0) = 2.5
    # i=2: a = ar = 0.5. v_2 = 0.5 * (2.5 + 30.0) = 16.25

    expected = [5.0, 2.5, 16.25]
    result = calculate_k2(x, delta, ar, ap)
    assert list(result) == expected

def test_calculate_k2_as_bad_as_old():
    """
    Test 'As-Bad-As-Old' case where ar=1 and ap=1.
    Virtual age should be the cumulative sum of intervals.
    """
    x = [10.0, 20.0, 30.0]
    delta = [1, 0, 1]
    ar = 1.0
    ap = 1.0

    expected = [10.0, 30.0, 60.0]
    result = calculate_k2(x, delta, ar, ap)
    assert list(result) == expected

def test_calculate_k2_as_good_as_new():
    """
    Test 'As-Good-As-New' case where ar=0 and ap=0.
    Virtual age should be 0 after each event.
    """
    x = [10.0, 20.0, 30.0]
    delta = [1, 0, 1]
    ar = 0.0
    ap = 0.0

    expected = [0.0, 0.0, 0.0]
    result = calculate_k2(x, delta, ar, ap)
    assert list(result) == expected

def test_calculate_k2_single_event():
    """
    Test calculation for a single event.
    """
    x = [10.0]
    delta = [1]
    ar = 0.5
    ap = 0.1

    expected = [5.0]
    result = calculate_k2(x, delta, ar, ap)
    assert list(result) == expected
