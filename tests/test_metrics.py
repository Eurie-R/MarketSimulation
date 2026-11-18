import numpy as np
import pytest

from marketabm import metrics


def test_hill_on_pareto() -> None:
    rng = np.random.default_rng(0)
    alpha = 3.0
    x = rng.pareto(alpha, 100_000) + 1.0  # classical Pareto, tail index alpha
    est = metrics.hill_alpha(x)
    assert abs(est - alpha) / alpha < 0.15


def test_kurtosis_of_normal() -> None:
    x = np.random.default_rng(1).standard_normal(200_000)
    assert abs(metrics.excess_kurtosis(x)) < 0.05


def test_acf_iid_within_band() -> None:
    x = np.random.default_rng(2).standard_normal(5000)
    a = metrics.acf(x, 50)
    band = metrics.acf_band(len(x))
    assert np.mean(np.abs(a[1:]) < band) >= 0.9


def test_max_drawdown_known() -> None:
    logp = np.array([0.0, 0.5, 0.2, 0.8, 0.1, 0.4, 0.9])
    assert metrics.max_drawdown(logp) == pytest.approx(0.7)


def test_period_returns_indices() -> None:
    price = np.exp(np.arange(30, dtype=float))
    r = metrics.period_returns(price, T_burn=10, interval=5)
    # ends at 9, 14, 19, 24, 29
    np.testing.assert_allclose(r, [5.0] * 4)


def test_nan_safety() -> None:
    assert np.isnan(metrics.vol(np.array([1.0])))
    assert np.isnan(metrics.rel_spread_mean(np.array([np.nan]), np.array([1.0])))
    assert np.isnan(metrics.hill_alpha(np.zeros(100)))
