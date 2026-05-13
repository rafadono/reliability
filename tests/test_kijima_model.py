import sys
from unittest.mock import MagicMock

# Mocking numpy and numba before importing app.kijima_model because they are not
# available in the current environment and cannot be installed.
# This allows testing the logic of calculate_virtual_age without physical dependencies.
if 'numpy' not in sys.modules:
    mock_np = MagicMock()
    mock_np.zeros_like.side_effect = lambda x, dtype=None: [0.0] * len(x)
    mock_np.asarray.side_effect = lambda x, dtype=None: x
    sys.modules['numpy'] = mock_np

if 'numba' not in sys.modules:
    mock_numba = MagicMock()
    mock_numba.njit.side_effect = lambda *args, **kwargs: (lambda f: f)
    sys.modules['numba'] = mock_numba

import pytest
from app.kijima_model import calculate_virtual_age

def test_calculate_virtual_age_invalid_model():
    """
    Test that calculate_virtual_age raises ValueError for an unsupported integer model_type.
    """
    x = [1.0, 2.0, 3.0]
    delta = [1.0, 0.0, 1.0]
    ar = 0.5
    ap = 0.8

    # model_type=3 is not 1 or 2
    with pytest.raises(ValueError) as excinfo:
        calculate_virtual_age(x, delta, ar, ap, model_type=3)
    assert "Invalid model_type: 3" in str(excinfo.value)

def test_calculate_virtual_age_invalid_model_list():
    """
    Test that calculate_virtual_age raises ValueError for an unsupported model_type inside a list.
    """
    x = [1.0, 2.0, 3.0]
    delta = [1.0, 0.0, 1.0]
    ar = 0.5
    ap = 0.8

    # model_type=[3] should be extracted as 3
    with pytest.raises(ValueError) as excinfo:
        calculate_virtual_age(x, delta, ar, ap, model_type=[3])
    assert "Invalid model_type: 3" in str(excinfo.value)
