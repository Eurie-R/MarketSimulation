import numpy as np
import pytest

from marketabm.agents import (
    CHART,
    FUND,
    NOISE,
    SKIP_NO_SIGNAL,
    SKIP_TAX,
    decide,
    init_agents,
    type_counts,
)
from marketabm.config import ModelConfig, with_overrides
from marketabm.rng import make_streams


@pytest.mark.parametrize("c", [0.0, 0.1, 0.25, 0.4, 0.55])
def test_type_counts_match_mix(baseline: ModelConfig, c: float) -> None:
    cfg = with_overrides(baseline, {"chartist_share": c, "N": 997})
    a = init_agents(cfg, make_streams(1).init)
    n_f, n_c, n_n = type_counts(cfg.N, cfg.mix.as_tuple())
    assert n_c == round(997 * cfg.mix.chartist) and n_n == round(997 * cfg.mix.noise)
    assert list(a.counts()) == [n_f, n_c, n_n]
    assert sum(a.counts()) == 997


def test_parameter_ranges(baseline: ModelConfig) -> None:
    a = init_agents(baseline, make_streams(3).init)
    for idx, prof in enumerate(baseline.profiles.by_index()):
        m = a.types == idx
        for name in ("g1", "g2", "n"):
            lo, hi = getattr(prof, name)
            vals = getattr(a, name)[m]
            assert vals.min() >= lo and vals.max() <= hi
    assert a.L.min() >= 5 and a.L.max() <= 50
    assert a.h.min() >= 10 and a.h.max() <= 100
    assert a.k.min() >= 0 and a.k.max() <= baseline.k_max
    noise = a.types == NOISE
    assert np.all(a.g1[noise] == 0) and np.all(a.g2[noise] == 0) and np.all(a.n[noise] == 1)
    assert (a.types == FUND).any() and (a.types == CHART).any()


def test_init_draws_independent_of_mix(baseline: ModelConfig) -> None:
    states = []
    for c in (0.0, 0.25, 0.55, 1.0):
        rng = make_streams(42).init
        init_agents(with_overrides(baseline, {"chartist_share": c}), rng)
        states.append(rng.bit_generator.state)
    assert all(s == states[0] for s in states)


def _decide(eps: float, tau: float, c_min: float = 0.0, **kw: float) -> tuple[int, int, int]:
    # Noise trader at p = f = 100: r_hat = eps, expected gain = |eps| * h (h = 50).
    lp = np.log(100.0)
    return decide(lp, lp, lp, 10, 0.0, 0.0, 1.0, 50, 0.0, eps, 100.0, tau, c_min, 0.01, **kw)


def test_decide_sides_and_price() -> None:
    side, ticks, qty = _decide(0.001, 0.0)
    assert side == 1 and ticks == round(100 * np.exp(0.05) / 0.01) and qty == 1
    side, ticks, qty = _decide(-0.001, 0.0)
    assert side == -1 and ticks == round(100 * np.exp(-0.05) / 0.01) and qty == 1


def test_decide_price_band() -> None:
    # forecast +5%, band 1%: buy limit at 100 * e^0.01; sell symmetric
    assert _decide(0.001, 0.0, price_band=0.01)[1] == round(100 * np.exp(0.01) / 0.01)
    assert _decide(-0.001, 0.0, price_band=0.01)[1] == round(100 * np.exp(-0.01) / 0.01)
    # inside the band: unchanged
    assert _decide(0.0001, 0.0, price_band=0.01)[1] == round(100 * np.exp(0.005) / 0.01)


def test_band_does_not_change_tax_filter() -> None:
    # gain 0.025 > band 0.01; the filter uses the unclipped gain
    assert _decide(0.0005, 0.02, price_band=0.01)[0] == 1
    assert _decide(0.0005, 0.03, price_band=0.01)[0] == SKIP_TAX


def test_decide_signal_size() -> None:
    # gain = 0.05, unit 0.01 -> 5 units; capped by size_max; at least 1
    assert _decide(0.001, 0.0, size_unit=0.01, size_max=20)[2] == 5
    assert _decide(0.001, 0.0, size_unit=0.01, size_max=3)[2] == 3
    assert _decide(0.00001, 0.0, size_unit=0.01, size_max=20)[2] == 1


def test_decide_tax_filter() -> None:
    # expected gain = 0.0005 * 50 = 0.025
    assert _decide(0.0005, 0.0)[0] == 1
    assert _decide(0.0005, 0.03)[0] == SKIP_TAX
    assert _decide(0.0005, 0.03, c_min=0.03)[0] == SKIP_NO_SIGNAL
    assert _decide(0.0, 0.0)[0] == SKIP_NO_SIGNAL
