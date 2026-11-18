import numpy as np
import pytest

from marketabm.config import ModelConfig, with_overrides
from marketabm.market import run


def test_short_run_shapes(small_stable: ModelConfig) -> None:
    r = run(small_stable, 11, debug=True)  # debug: <= 1 live order per agent, conservation
    T = small_stable.T  # noqa: N806
    for name in ("price", "fundamental", "best_bid", "best_ask", "volume", "tax_revenue_cum"):
        assert getattr(r, name).shape == (T,)
    for name in (
        "fills_by_type",
        "aggressive_fills_by_type",
        "skipped_tax_by_type",
        "skipped_no_signal_by_type",
        "active_by_type",
    ):
        assert getattr(r, name).shape == (T, 3)
    n_snap = T // small_stable.return_interval
    assert r.depth_bid.shape == r.depth_ask.shape == r.book_size.shape == (n_snap,)
    assert r.active_by_type.sum(axis=1).tolist() == [small_stable.M] * T
    assert len(r.period_returns()) == (T - small_stable.T_burn) // small_stable.return_interval
    s = r.summary()
    assert np.isfinite(s["vol"]) and s["volume_total"] > 0


def test_prices_positive_on_tick_grid(small_stable: ModelConfig) -> None:
    r = run(small_stable, 12)
    assert np.all(r.price > 0)
    ticks = r.price / small_stable.tick
    assert np.allclose(ticks, np.round(ticks), atol=1e-9)


def test_fundamentalists_track_value(baseline: ModelConfig) -> None:
    cfg = with_overrides(
        baseline,
        {
            "T": 6000,
            "T_burn": 1000,
            "N": 200,
            "sigma_eps": 1e-7,
            "mix": {"fundamentalist": 1.0, "chartist": 0.0, "noise": 0.0},
        },
    )
    s = run(cfg, 5).summary()
    assert s["mispricing"] < 0.02


@pytest.mark.parametrize("c", [0.10, 0.25, 0.40, 0.55])
def test_bounded_at_all_e2_chartist_shares(baseline: ModelConfig, c: float) -> None:
    cfg = with_overrides(baseline, {"T": 6000, "T_burn": 1000, "N": 300, "chartist_share": c})
    r = run(cfg, 2)
    assert np.isfinite(r.price).all()
    assert r.summary()["mispricing"] < 1.0  # no runaway bubble (price within ~e of value)


def test_original_rule_diverges(baseline: ModelConfig) -> None:
    """Documents why the price band exists: v1 rule (no band, unit size, tau_f = 100)."""
    cfg = with_overrides(
        baseline,
        {"T": 2000, "T_burn": 500, "N": 100, "price_band": None, "order_size": "unit",
         "tau_f": 100},
    )  # fmt: skip
    with pytest.raises(OverflowError):
        run(cfg, 1)


def test_order_sizes_within_limits(small_stable: ModelConfig) -> None:
    r = run(small_stable, 4)
    # More units than participations/2 means multi-unit orders occurred.
    assert r.volume.sum() > r.fills_by_type.sum() / 2
    unit = run(with_overrides(small_stable, {"order_size": "unit"}), 4)
    assert unit.volume.sum() == unit.fills_by_type.sum() / 2


def test_live_matches_run(small_stable: ModelConfig) -> None:
    from marketabm.market import live

    r = run(small_stable, seed=3)
    frames = list(live(small_stable, seed=3, every=250))
    assert [fr.t for fr in frames] == list(range(249, small_stable.T, 250))
    last = frames[-1]
    assert np.array_equal(last.price, r.price)
    assert np.array_equal(last.fills_by_type, r.fills_by_type)
    assert last.tax_revenue == r.tax_revenue_cum[-1]
    for fr in frames:
        if fr.bids and fr.asks:
            assert fr.bids[0][0] < fr.asks[0][0]
            assert [p for p, _ in fr.bids] == sorted((p for p, _ in fr.bids), reverse=True)


def test_live_window_is_tail(small_stable: ModelConfig) -> None:
    from marketabm.market import Simulation

    r = run(small_stable, seed=4)
    sim = Simulation(small_stable, seed=4)
    sim.advance(700)
    fr = sim.frame(window=200)
    assert (fr.t_start, fr.t) == (500, 699)
    assert np.array_equal(fr.price, r.price[500:700])
    assert np.array_equal(fr.active_by_type, r.active_by_type[500:700])


def test_vol_feedback_off_and_gamma_zero_match_v11(small_stable: ModelConfig) -> None:
    r_off = run(small_stable, seed=5)
    on0 = with_overrides(small_stable, {"vol_feedback": {"enabled": True, "gamma": 0.0}})
    r0 = run(on0, seed=5)
    assert np.array_equal(r_off.price, r0.price)
    assert np.all(r_off.vol_mult == 1.0) and np.all(r0.vol_mult == 1.0)


def test_vol_feedback_changes_path_within_cap(small_stable: ModelConfig) -> None:
    cfg = with_overrides(small_stable, {"vol_feedback": {"enabled": True, "gamma": 2.0,
                                                         "cap": 3.0}})  # fmt: skip
    r = run(cfg, seed=5)
    assert not np.array_equal(r.price, run(small_stable, seed=5).price)
    assert r.vol_mult[0] == 1.0
    assert np.all((r.vol_mult >= 1 / 3.0 - 1e-12) & (r.vol_mult <= 3.0 + 1e-12))
    assert r.vol_mult.std() > 0


def test_vol_feedback_state_tracks_ratio() -> None:
    from marketabm.config import VolFeedback
    from marketabm.market import VolFeedbackState

    vf = VolFeedbackState(VolFeedback(enabled=True, gamma=1.0, fast_halflife=5,
                                      slow_halflife=500, cap=10.0))  # fmt: skip
    for _ in range(2000):
        vf.update(0.001)
    assert abs(vf.mult - 1.0) < 1e-9  # constant volatility -> ratio 1
    for _ in range(50):
        vf.update(0.004)
    assert vf.mult > 2.0  # a burst raises fast vol well above slow vol
