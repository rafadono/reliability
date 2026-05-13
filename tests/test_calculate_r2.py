import math
from unittest.mock import patch
import app.tests as mod

class MockArray:
    def __init__(self, data):
        self.data = list(data)

    def mean(self):
        if not self.data:
            return 0.0
        return sum(self.data) / len(self.data)

    def __sub__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a - b for a, b in zip(self.data, other.data)])
        return MockArray([a - other for a in self.data])

    def __pow__(self, exponent):
        return MockArray([a ** exponent for a in self.data])

    def __iter__(self):
        return iter(self.data)

def mock_np_sum(iterable, *args, **kwargs):
    if hasattr(iterable, '__iter__'):
        return sum(iterable)
    return iterable

@patch('app.tests.np.sum', side_effect=mock_np_sum)
def test_calculate_r2_perfect_prediction(mock_sum):
    x = MockArray([1.0, 2.0, 3.0, 4.0, 5.0])
    V = MockArray([1.0, 2.0, 3.0, 4.0, 5.0])

    r2 = mod.calculate_r2(x, V)

    assert r2 == 1.0

@patch('app.tests.np.sum', side_effect=mock_np_sum)
def test_calculate_r2_zero_prediction_ss_res_equals_ss_tot(mock_sum):
    x = MockArray([1.0, 2.0, 3.0, 4.0, 5.0])
    mean_val = x.mean()
    V = MockArray([mean_val] * 5)

    r2 = mod.calculate_r2(x, V)

    assert r2 == 0.0

@patch('app.tests.np.sum', side_effect=mock_np_sum)
def test_calculate_r2_general_case(mock_sum):
    x = MockArray([2.0, 4.0, 6.0])
    V = MockArray([3.0, 4.0, 5.0])

    r2 = mod.calculate_r2(x, V)
    assert math.isclose(r2, 0.75)

@patch('app.tests.np.sum', side_effect=mock_np_sum)
def test_calculate_r2_negative(mock_sum):
    x = MockArray([2.0, 4.0, 6.0])
    V2 = MockArray([0.0, 8.0, 0.0])

    r2 = mod.calculate_r2(x, V2)
    assert r2 == -6.0
