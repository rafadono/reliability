from app.kijima_model import calculate_k2

def test_calculate_k2_happy_path():
    x = [10.0, 20.0, 30.0]
    delta = [1, 0, 1]
    ar = 0.5
    ap = 0.1

    expected = [5.0, 2.5, 16.25]
    result = calculate_k2(x, delta, ar, ap)
    assert list(result) == expected

def test_calculate_k2_as_bad_as_old():
    x = [10.0, 20.0, 30.0]
    delta = [1, 0, 1]
    ar = 1.0
    ap = 1.0

    expected = [10.0, 30.0, 60.0]
    result = calculate_k2(x, delta, ar, ap)
    assert list(result) == expected

def test_calculate_k2_as_good_as_new():
    x = [10.0, 20.0, 30.0]
    delta = [1, 0, 1]
    ar = 0.0
    ap = 0.0

    expected = [0.0, 0.0, 0.0]
    result = calculate_k2(x, delta, ar, ap)
    assert list(result) == expected

def test_calculate_k2_single_event():
    x = [10.0]
    delta = [1]
    ar = 0.5
    ap = 0.1

    expected = [5.0]
    result = calculate_k2(x, delta, ar, ap)
    assert list(result) == expected
