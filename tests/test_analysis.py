import numpy as np
import pandas as pd
import pytest

from marketabm import analysis
from marketabm.config import mix_from_chartist_share


def test_mix_from_chartist_share_values() -> None:
    m = mix_from_chartist_share(0.25, {"fundamentalist": 0.45, "chartist": 0.25, "noise": 0.30})
    assert m == pytest.approx({"fundamentalist": 0.45, "chartist": 0.25, "noise": 0.30})
    assert sum(m.values()) == pytest.approx(1.0)


def _synthetic_runs(effect: float, n_seeds: int = 40, seed: int = 0) -> pd.DataFrame:
    """vol = seed effect + noise, plus ``effect`` per unit of tau (in %); shares 0.25 / 0.4."""
    rng = np.random.default_rng(seed)
    rows = []
    for share in (0.25, 0.40):
        base = rng.normal(1.0, 0.3, size=n_seeds)  # seed-level effect shared across tau
        for tau in (0.0, 0.001, 0.006):
            noise = rng.normal(0, 0.02, size=n_seeds)
            for s in range(n_seeds):
                rows.append({"chartist_share": share, "tau": tau, "seed": 1000 + s,
                             "vol": base[s] + effect * tau * 100 + noise[s]})  # fmt: skip
    return pd.DataFrame(rows)


def test_paired_differences_recover_effect() -> None:
    runs = _synthetic_runs(effect=-0.05)
    d = analysis.paired_differences(runs, "vol", ["chartist_share"])
    assert set(d["tau"]) == {0.001, 0.006}
    for _, r in d.iterrows():
        truth = -0.05 * r["tau"] * 100
        assert r["ci_low"] <= truth <= r["ci_high"]
        assert abs(r["mean"] - truth) < 0.01
        assert r["n"] == 40


def test_pairing_removes_seed_variance() -> None:
    # The seed effect (sd 0.3) is far larger than the tax effect; pairing must still find it.
    runs = _synthetic_runs(effect=-0.05)
    d = analysis.paired_differences(runs, "vol", ["chartist_share"])
    assert (d.loc[d["tau"] == 0.006, "ci_high"] < 0).all()


def test_ph_cut_effect_sign() -> None:
    runs = _synthetic_runs(effect=-0.05)
    ph = analysis.ph_cut_effect(runs, ["vol"], ["chartist_share"])
    truth = -0.05 * 0.1 - (-0.05 * 0.6)  # metric(0.1%) - metric(0.6%) = +0.025
    assert ((ph["ci_low"] <= truth) & (truth <= ph["ci_high"])).all()


def test_relative_paired_difference() -> None:
    runs = pd.DataFrame({"tau": [0.0, 0.01, 0.0, 0.01], "seed": [1, 1, 2, 2],
                         "x": [10.0, 5.0, 20.0, 10.0]})  # fmt: skip
    d = analysis.paired_differences(runs, "x", [], relative=True)
    assert d["mean"].iloc[0] == pytest.approx(-0.5)


def test_t_interval_and_cell_means() -> None:
    m, lo, hi, n = analysis.t_interval([1.0, 2.0, 3.0, np.nan])
    assert (m, n) == (2.0, 3) and lo < 2 < hi
    runs = _synthetic_runs(effect=0.0)
    cm = analysis.cell_means(runs, ["chartist_share", "tau"], ["vol"])
    assert len(cm) == 6 and (cm["n"] == 40).all()


def test_extreme_moves_uses_paired_baseline() -> None:
    runs = pd.DataFrame({"run_id": [0, 1], "tau": [0.0, 0.006], "seed": [7, 7],
                         "N": [250, 250], "vol": [1.0, 2.0]})  # fmt: skip
    returns = {0: np.array([0.0, 6.0, -7.0, 1.0]), 1: np.array([5.5, 4.0, 10.0, 0.0])}
    ex = analysis.extreme_moves(runs, returns, ["N"])
    assert ex.tolist() == [2.0, 2.0]  # both counted against vol_ref = 1.0 of the tau = 0 run


def test_h3_verdict_on_synthetic() -> None:
    rows = []
    for tau in (0.0, 0.006):
        for s in range(20):
            e = 0.01 * s
            rows.append(
                {
                    "N": 250,
                    "tau": tau,
                    "seed": s,
                    "vol": 1 + e + 2 * tau,
                    "rel_spread_mean": 0.01 + e + tau,
                    "depth_mean": 100 - e - 1000 * tau,
                }
            )
    v = analysis.check_h3(pd.DataFrame(rows))
    assert v.verdict == "supported"
