import sys
import math
from unittest.mock import MagicMock, patch

# In order to test the behavior, let's just use Python's built-in lists
# and mock numpy and scipy such that they return lists/floats instead of complex mock arrays.

mock_np = MagicMock()

def np_exp_mock(val):
    if isinstance(val, list):
        return [math.exp(v) for v in val]
    return math.exp(val)

def np_sum_mock(val):
    if isinstance(val, list):
        return sum(val)
    if hasattr(val, '__iter__'):
        return sum(val)
    return val

mock_np.exp = MagicMock(side_effect=np_exp_mock)
mock_np.sum = MagicMock(side_effect=np_sum_mock)
mock_np.inf = float('inf')
mock_np.sort = MagicMock(side_effect=lambda x: sorted(x))
mock_np.empty = MagicMock(side_effect=lambda n: [0.0] * n)
mock_np.log = MagicMock(side_effect=math.log)

sys.modules["numpy"] = mock_np
sys.modules["scipy"] = MagicMock()
mock_scipy_stats = MagicMock()
sys.modules["scipy.stats"] = mock_scipy_stats
mock_scipy_integrate = MagicMock()
sys.modules["scipy.integrate"] = mock_scipy_integrate
sys.modules["reliability"] = MagicMock()
sys.modules["reliability.Nonparametric"] = MagicMock()
mock_app_kijima_model = MagicMock()
sys.modules["app.kijima_model"] = mock_app_kijima_model

from app.tests import (
    kolmogorov_smirnov_test,
    mean_residual_life,
    r2_mrl,
    ks_test_weibull_pit,
    ks_test_kijima_pit,
)

# Replace the module imports inside app.tests that were already resolved to the original mocks
import app.tests
app.tests.np = mock_np
app.tests.kstest = mock_scipy_stats.kstest
app.tests.quad = mock_scipy_integrate.quad
app.tests.pdf = mock_app_kijima_model.pdf
app.tests.calculate_virtual_age = mock_app_kijima_model.calculate_virtual_age


def test_kolmogorov_smirnov_test():
    x = [1, 2, 3]
    beta = 2.0
    eta = 1.5
    mock_scipy_stats.kstest.return_value = "kstest_result"

    result = kolmogorov_smirnov_test(x, beta, eta)

    mock_scipy_stats.kstest.assert_called_once_with(x, 'weibull_min', args=(beta, 0, eta))
    assert result == "kstest_result"


def test_mean_residual_life():
    v_prev = 1.0
    beta = 2.0
    eta = 1.5

    # quad returns a tuple (integral, error)
    mock_scipy_integrate.quad.return_value = (42.0, 0.1)

    result = mean_residual_life(v_prev, beta, eta)

    mock_scipy_integrate.quad.assert_called_once()
    integrand = mock_scipy_integrate.quad.call_args[0][0]

    mock_app_kijima_model.pdf.return_value = 5.0
    val = integrand(2.0)

    mock_app_kijima_model.pdf.assert_called_once_with(2.0, v_prev, beta, eta)
    assert val == 2.0 * 5.0
    assert result == 42.0


def test_r2_mrl(monkeypatch):
    class DummyArray:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(data)

        def mean(self):
            return sum(self.data) / len(self.data) if self.data else 0.0

        def __sub__(self, other):
            if hasattr(other, 'data'):
                return [a - b for a, b in zip(self.data, other.data)]
            if isinstance(other, list):
                return [a - b for a, b in zip(self.data, other)]
            return [a - other for a in self.data]

        def __rsub__(self, other):
             return [other - a for a in self.data]

        def __getitem__(self, idx):
            return self.data[idx]

    # For math operations on the results of __sub__ (which are lists)
    def dummy_sum(iterable):
        if hasattr(iterable, 'data'):
            return sum(iterable.data)
        return sum(iterable)

    mock_np.sum.side_effect = dummy_sum

    x = DummyArray([10.0, 20.0, 30.0])
    delta = [1, 0, 1]
    ar = 0.5
    ap = 0.5
    beta = 2.0
    eta = 1.5
    model_type = 1

    mock_app_kijima_model.calculate_virtual_age.return_value = [5.0, 10.0, 15.0]

    mock_mrl = MagicMock()
    mock_mrl.side_effect = [100.0, 200.0, 300.0]
    monkeypatch.setattr("app.tests.mean_residual_life", mock_mrl)

    # We also need to patch the list comprehension squared operations since __sub__ returns list
    # Actually, we can just replace np.sum with a custom lambda that computes square sums
    def np_sum_patch(val):
        # The input here would be an expression like (y_obs - y_obs.mean())**2
        # But wait, **2 on list will fail in python. Let's make DummyArray __sub__ return another DummyArray
        # so it supports **2
        pass

    class AdvancedArray:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(data)

        def mean(self):
            return sum(self.data) / len(self.data) if self.data else 0.0

        def __sub__(self, other):
            if hasattr(other, 'data'):
                return AdvancedArray([a - b for a, b in zip(self.data, other.data)])
            if isinstance(other, list):
                return AdvancedArray([a - b for a, b in zip(self.data, other)])
            return AdvancedArray([a - other for a in self.data])

        def __pow__(self, other):
            return AdvancedArray([a ** other for a in self.data])

        def __getitem__(self, idx):
            return self.data[idx]

        def __iter__(self):
            return iter(self.data)

    x = AdvancedArray([10.0, 20.0, 30.0])

    mock_np.empty.return_value = [0.0, 0.0, 0.0]

    result = r2_mrl(x, delta, ar, ap, beta, eta, model_type)

    mock_app_kijima_model.calculate_virtual_age.assert_called_once_with(x, delta, ar, ap, model_type)

    # mean_residual_life is called for i=0..n-1
    assert mock_mrl.call_count == 3
    mock_mrl.assert_any_call(0.0, beta, eta)
    mock_mrl.assert_any_call(5.0, beta, eta)
    mock_mrl.assert_any_call(10.0, beta, eta)

    y_pred = [100.0, 200.0, 300.0]
    y_obs = x.data
    mean_obs = sum(y_obs)/len(y_obs)
    ss_tot = sum((val - mean_obs)**2 for val in y_obs)
    ss_res = sum((obs - pred)**2 for obs, pred in zip(y_obs, y_pred))
    expected_r2 = 1 - ss_res/ss_tot

    assert math.isclose(result, expected_r2)

def test_ks_test_weibull_pit():
    class PitArray:
        def __init__(self, data):
            self.data = list(data)

        def __truediv__(self, other):
            return PitArray([v / other for v in self.data])

        def __pow__(self, other):
            return PitArray([v ** other for v in self.data])

        def __neg__(self):
            return PitArray([-v for v in self.data])

        def __rsub__(self, other):
            return PitArray([other - v for v in self.data])

        def __iter__(self):
            return iter(self.data)

    x = PitArray([1.0, 2.0])
    beta = 2.0
    eta = 1.5
    mock_scipy_stats.kstest.reset_mock()
    mock_scipy_stats.kstest.return_value = "weibull_pit_result"

    mock_np.exp.side_effect = lambda a: PitArray([math.exp(v) for v in a.data])

    result = ks_test_weibull_pit(x, beta, eta)

    expected_F = [1 - math.exp(-(val/eta)**beta) for val in x.data]

    call_args = mock_scipy_stats.kstest.call_args[0]
    assert call_args[0].data == expected_F
    assert call_args[1] == 'uniform'

    assert result == "weibull_pit_result"

def test_ks_test_kijima_pit():
    class PitArray2:
        def __init__(self, data):
            self.data = list(data)

        def __add__(self, other):
            if isinstance(other, PitArray2):
                return PitArray2([a + b for a, b in zip(self.data, other.data)])
            return PitArray2([a + other for a in self.data])

        def __radd__(self, other):
            return PitArray2([other + a for a in self.data])

        def __truediv__(self, other):
            return PitArray2([v / other for v in self.data])

        def __pow__(self, other):
            return PitArray2([v ** other for v in self.data])

        def __sub__(self, other):
            if isinstance(other, PitArray2):
                return PitArray2([a - b for a, b in zip(self.data, other.data)])
            return PitArray2([a - other for a in self.data])

        def __rsub__(self, other):
            return PitArray2([other - v for v in self.data])

        def __getitem__(self, idx):
            if isinstance(idx, list) or isinstance(idx, PitArray2):
                bools = idx.data if hasattr(idx, 'data') else idx
                return PitArray2([self.data[i] for i, b in enumerate(bools) if b])
            return self.data[idx]

        def __eq__(self, other):
            return PitArray2([v == other for v in self.data])

    x = PitArray2([30.0, 10.0, 20.0])
    delta = PitArray2([1, 1, 0])
    beta = 2.0
    eta = 1.5
    ar = 0.5
    ap = 0.5
    model_type = 1

    mock_scipy_stats.kstest.reset_mock()
    mock_scipy_stats.kstest.return_value = "kijima_pit_result"

    mock_np.sort.side_effect = lambda a: PitArray2(sorted(a.data))
    mock_app_kijima_model.calculate_virtual_age.return_value = [5.0, 10.0, 15.0]

    def np_exp_patch(val):
        if hasattr(val, 'data'):
            return PitArray2([math.exp(v) for v in val.data])
        return math.exp(val)
    mock_np.exp.side_effect = np_exp_patch

    result = ks_test_kijima_pit(x, delta, beta, eta, ar, ap, model_type)

    sorted_x = sorted(x.data)

    expected_S = [math.exp((15.0/eta)**beta - ((15.0 + val)/eta)**beta) for val in sorted_x]
    expected_F = [1 - val for val in expected_S]
    expected_F_fail = [f for i, f in enumerate(expected_F) if delta.data[i] == 1]

    call_args = mock_scipy_stats.kstest.call_args[0]
    # Almost equal check
    assert all(math.isclose(a, b) for a, b in zip(call_args[0].data, expected_F_fail))
    assert call_args[1] == 'uniform'

    assert result == "kijima_pit_result"
