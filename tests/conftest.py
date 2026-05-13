import sys
import math
from unittest.mock import MagicMock

def pytest_configure():
    mock_np = MagicMock()

    def mock_sum(iterable, *args, **kwargs):
        if hasattr(iterable, '__iter__'):
            return sum(iterable)
        return iterable
    mock_np.sum.side_effect = mock_sum
    mock_np.log.side_effect = math.log

    def mock_zeros_like(a, dtype=None):
        return [0.0] * len(a)
    mock_np.zeros_like.side_effect = mock_zeros_like
    mock_np.asarray.side_effect = lambda x, dtype=None: x

    sys.modules["numpy"] = mock_np

    mock_numba = MagicMock()
    mock_numba.njit.side_effect = lambda *args, **kwargs: (lambda f: f)
    sys.modules["numba"] = mock_numba

    sys.modules["scipy"] = MagicMock()
    sys.modules["scipy.stats"] = MagicMock()
    sys.modules["scipy.integrate"] = MagicMock()
    sys.modules["reliability"] = MagicMock()
    sys.modules["reliability.Nonparametric"] = MagicMock()
