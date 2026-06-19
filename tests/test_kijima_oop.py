import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaincc, gamma as scipy_gamma
from src.reliability_analysis.analysis.kijima_model import (
    KijimaModelI,
    KijimaModelII,
    KijimaModelITD,
    KijimaModelIITD,
    _safe_exp_gamma,
)


def test_kijima_oop_instantiation():
    """Verify that Kijima model subclasses instantiate and hold parameters."""
    m1 = KijimaModelI(beta=1.5, eta=500.0, ar=0.2, ap=0.8)
    assert m1.beta == 1.5
    assert m1.eta == 500.0
    assert m1.ar == 0.2
    assert m1.ap == 0.8
    assert m1.model_type == 1
    assert m1.model_name == "Kijima I"

    m3 = KijimaModelITD(beta=1.2, eta=400.0, ar=0.3, ap=0.7, br=0.002, bp=-0.001)
    assert m3.br == 0.002
    assert m3.bp == -0.001
    assert m3.model_type == 3


def test_kijima_oop_virtual_age_consistency():
    """Verify that all four subclasses produce finite, non-negative virtual ages."""
    x = np.array([100.0, 150.0, 200.0, 80.0])
    delta = np.array([1.0, 0.0, 1.0, 0.0])

    models = [
        KijimaModelI(1.5, 500.0, 0.2, 0.8),
        KijimaModelII(1.5, 500.0, 0.2, 0.8),
        KijimaModelITD(1.5, 500.0, 0.2, 0.8, 0.001, -0.001),
        KijimaModelIITD(1.5, 500.0, 0.2, 0.8, 0.001, -0.001),
    ]
    for model in models:
        V = model.virtual_age(x, delta)
        assert V.shape == x.shape
        assert np.all(np.isfinite(V))
        assert np.all(V >= 0.0)


def test_kijima_oop_distribution_functions():
    """Verify standard probability functions."""
    m1 = KijimaModelI(beta=1.5, eta=500.0, ar=0.2, ap=0.8)
    t = np.array([0.0, 10.0, 50.0, 100.0])
    V = 50.0

    sf = m1.sf(t, V)
    rel = m1.reliability(t, V)
    np.testing.assert_array_equal(sf, rel)
    assert sf[0] == pytest.approx(1.0)
    assert np.all(sf >= 0.0) and np.all(sf <= 1.0)

    cdf = m1.cdf(t, V)
    np.testing.assert_array_almost_equal(cdf + sf, np.ones_like(t))

    pdf = m1.pdf(t, V)
    haz = m1.hazard(t, V)
    # hazard = pdf / sf
    np.testing.assert_array_almost_equal(haz * sf, pdf)


def test_kijima_oop_analytical_mtbf_and_std():
    """Verify that analytical mean/std match numerical integration."""
    models = [
        KijimaModelI(beta=1.8, eta=600.0, ar=0.1, ap=0.9),
        KijimaModelII(beta=0.8, eta=300.0, ar=0.4, ap=0.5),
        KijimaModelITD(beta=1.5, eta=500.0, ar=0.2, ap=0.8, br=0.001, bp=-0.002),
        KijimaModelIITD(beta=2.2, eta=800.0, ar=0.1, ap=0.7, br=0.0005, bp=0.0001),
    ]

    for model in models:
        for V in [0.0, 50.0, 200.0]:
            mtbf_num, _ = quad(lambda t: model.sf(t, V), 0, np.inf)
            mtbf_ana = model.mean(V)
            assert mtbf_ana == pytest.approx(mtbf_num, rel=1e-5)

            E2_num, _ = quad(lambda t: 2.0 * t * model.sf(t, V), 0, np.inf)
            var_num = E2_num - mtbf_num**2
            std_num = np.sqrt(var_num) if var_num > 0 else 0.0
            std_ana = model.std(V)
            assert std_ana == pytest.approx(std_num, rel=1e-5)


def test_kijima_oop_calculate_curves():
    """Verify that calculate_curves outputs the correct shapes and values."""
    m1 = KijimaModelI(beta=1.5, eta=500.0, ar=0.2, ap=0.8)
    x = np.array([100.0, 150.0, 200.0])
    delta = np.array([1.0, 0.0, 1.0])
    t_grid = np.linspace(0, 450.0, 50)

    curves = m1.calculate_curves(x, delta, t_grid)
    assert set(curves.keys()) >= {"t", "R", "failure_rate", "pdf", "V", "T"}
    assert len(curves["t"]) == 65
    assert len(curves["R"]) == 65
    assert len(curves["V"]) == 3
    assert len(curves["T"]) == 4
    assert curves["R"][0] == pytest.approx(1.0)


def test_kijima_oop_ks_test_pit():
    """Verify that model.ks_test_pit returns valid statistic and p-value."""
    x = np.array([100.0, 150.0, 200.0, 80.0])
    delta = np.array([1.0, 0.0, 1.0, 1.0])
    m1 = KijimaModelI(beta=1.5, eta=500.0, ar=0.2, ap=0.8)
    stat, pval = m1.ks_test_pit(x, delta)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= pval <= 1.0


def test_kijima_oop_auc_improvement():
    """Verify auc_improvement is finite and bounded."""
    m1 = KijimaModelI(beta=1.5, eta=500.0, ar=0.2, ap=0.8)
    auc = m1.auc_improvement(Vn=50.0, t_max=1000.0)
    assert np.isfinite(auc)


def test_safe_exp_gamma():
    """Verify _safe_exp_gamma matches scipy for small x and is finite for large x."""
    a = 0.5
    for x in [0.5, 5.0, 10.0, 49.0]:
        expected = np.exp(x) * gammaincc(a, x) * scipy_gamma(a)
        assert _safe_exp_gamma(a, x) == pytest.approx(expected, rel=1e-12)

    # Large x: asymptotic path should be close to the exact value
    expected_51 = np.exp(51.0) * gammaincc(a, 51.0) * scipy_gamma(a)
    assert _safe_exp_gamma(a, 51.0) == pytest.approx(expected_51, rel=1e-4)

    # Overflow regime: result must be finite and positive
    val = _safe_exp_gamma(a, 1000.0)
    assert np.isfinite(val)
    assert val > 0.0


def test_kijima_large_virtual_age():
    """Verify mean/variance/std are finite under large virtual age V."""
    m1 = KijimaModelI(beta=1.5, eta=100.0, ar=0.5, ap=0.5)
    # (V/eta)**beta = (2000/100)**1.5 ≈ 89.4  (> 50, triggers asymptotic path)
    mean_val = m1.mean(2000.0)
    var_val = m1.variance(2000.0)
    std_val = m1.std(2000.0)
    assert np.isfinite(mean_val) and mean_val > 0.0
    assert np.isfinite(var_val) and var_val > 0.0
    assert np.isfinite(std_val)
