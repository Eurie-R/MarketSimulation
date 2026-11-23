"""Tests for empirical.py and calibration.py (E1)."""

from pathlib import Path

import numpy as np
import pandas as pd

from marketabm import calibration as cal
from marketabm import empirical
from marketabm.stylized_facts import MOMENT_NAMES

ROOT = Path(__file__).resolve().parents[1]


def test_clean_returns_drops_holidays_and_missing() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
            "Close": [100.0, np.nan, 110.0, 110.0, 121.0],
            "Volume": [5, 5, 5, 0, 5],
        }
    )
    r = empirical.clean_returns(df)
    assert np.allclose(r.to_numpy(), [np.log(1.1), np.log(1.1)])


def test_block_bootstrap_shape_and_weights() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_t(4, size=1000)
    boot = empirical.block_bootstrap_moments(r, np.random.default_rng(1), block=20, n_boot=50)
    assert boot.shape == (50, len(MOMENT_NAMES))
    w = empirical.moment_weights(boot)
    assert np.all(np.isfinite(w)) and np.all(w > 0)


def test_block_bootstrap_is_deterministic() -> None:
    r = np.random.default_rng(0).normal(size=500)
    a = empirical.block_bootstrap_moments(r, np.random.default_rng(7), n_boot=10)
    b = empirical.block_bootstrap_moments(r, np.random.default_rng(7), n_boot=10)
    assert np.array_equal(a, b, equal_nan=True)


def test_e1_grid_has_1728_points() -> None:
    cfg = cal.load_calibration_config(ROOT / "configs" / "e1_calibration.yaml")
    assert len(cal.grid_points(cfg.grid)) == 1728
    assert "vol_feedback_gamma" in cfg.grid
    assert len(cfg.seeds_stage1) == 3 and len(cfg.seeds_stage2) == 30


def test_point_overrides_maps_chartist_g2(baseline) -> None:
    from marketabm.config import with_overrides

    point = {"tau_f": 25, "chartist_g2": [0.4, 0.6]}
    cfg = with_overrides(baseline, cal.point_overrides(point))
    assert cfg.tau_f == 25
    assert tuple(cfg.profiles.chartist.g2) == (0.4, 0.6)
    assert tuple(cfg.profiles.chartist.g1) == tuple(baseline.profiles.chartist.g1)


def test_distance() -> None:
    assert cal.distance(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([2.0, 0.5])) == 4.0


def _means_row(**kw: float) -> pd.Series:
    base = {
        "excess_kurtosis": 2.0,
        "hill_alpha": 3.5,
        "acf_r_lag1": 0.0,
        "acf_band": 0.04,
        **{c: 0.1 for c in cal.ACF_COLS},
        **{c: 0.0 for c in cal.ACF_R_COLS},
    }
    base.update(kw)
    return pd.Series(base)


def test_checks_from_means() -> None:
    assert all(cal.checks_from_means(_means_row()).values())
    assert not cal.checks_from_means(_means_row(acf_r_lag1=-0.4))["|acf1|<0.1"]
    assert not cal.checks_from_means(_means_row(acf_abs_20=0.01))["acf|R|>band 1-20"]
    assert not cal.checks_from_means(_means_row(hill_alpha=7.0))["2<hill<6"]
    assert not cal.checks_from_means(_means_row(acf_r_2=-0.69))["|acfR 2-5|<0.1"]


def test_choose_picks_lowest_passing() -> None:
    df = pd.DataFrame({"D": [1.0, 2.0, 3.0], "passes": [False, True, True]}, index=[5, 6, 7])
    assert cal.choose(df) == 6
    assert cal.choose(df.assign(passes=False)) is None


def test_simulate_point_returns_moments(small_stable) -> None:
    row = cal.simulate_point(small_stable, {"tau_f": 25, "chartist_g2": [0.4, 0.6]}, seed=1)
    assert row["ok"]
    assert all(np.isfinite(row[m]) for m in MOMENT_NAMES)


def test_source_roundtrip(tmp_path: Path) -> None:
    r = pd.Series([0.01, -0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    p = empirical.save_source("EPHE (Yahoo Finance)", r, tmp_path / "source.json")
    meta = empirical.load_source(p)
    assert meta["source"] == "EPHE (Yahoo Finance)" and meta["n_returns"] == 2
    assert empirical.load_source(tmp_path / "missing.json") == {"source": "unknown"}


def test_point_overrides_maps_vol_feedback(baseline) -> None:
    from marketabm.config import with_overrides

    on = with_overrides(baseline, cal.point_overrides({"vol_feedback_gamma": 2}))
    off = with_overrides(baseline, cal.point_overrides({"vol_feedback_gamma": 0}))
    assert on.vol_feedback.enabled and on.vol_feedback.gamma == 2.0
    assert not off.vol_feedback.enabled
