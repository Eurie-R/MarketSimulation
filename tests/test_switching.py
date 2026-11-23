"""Strategy switching and the E3 plumbing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketabm.agents import CHART, FUND, NOISE, init_agents
from marketabm.config import ExperimentConfig, ModelConfig, with_overrides
from marketabm.market import Simulation, run
from marketabm.rng import make_streams
from marketabm.switching import logit_probs, switching_round, type_profits


@pytest.fixture(scope="module")
def sw_cfg(small_stable: ModelConfig) -> ModelConfig:
    return with_overrides(small_stable, {"beta": 5.0, "switching": {"S": 250}})


def test_logit_probs_stable_and_neutral() -> None:
    np.testing.assert_allclose(logit_probs(np.array([0.3, -0.2]), 0.0), [0.5, 0.5])
    p = logit_probs(np.array([1e6, 0.0]), 10.0)  # would overflow without the max shift
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0) and p[0] == pytest.approx(1.0)
    p = logit_probs(np.array([0.1, 0.0]), 2.0)
    assert p[0] == pytest.approx(1 / (1 + np.exp(-0.2)))


def test_type_profits_scaled_and_empty_type_zero() -> None:
    types = np.array([FUND, FUND, NOISE], dtype=np.int8)
    pi = type_profits(np.array([3.0, 5.0, 9.0]), np.array([1.0, 1.0, 0.0]), types, p0=2.0)
    np.testing.assert_allclose(pi, [1.5, 0.0])  # FUND mean gain 3 / p0; no chartists -> 0


def test_switching_round_rules(small_stable: ModelConfig) -> None:
    cfg = small_stable
    agents = init_agents(cfg, make_streams(5).init)
    before = {k: getattr(agents, k).copy() for k in ("types", "g1", "g2", "n", "L", "h", "k")}
    rng = np.random.default_rng(0)
    w_now = np.where(agents.types == CHART, 50.0, 0.0)  # chartists did far better
    changed, pi = switching_round(agents, w_now, np.zeros(cfg.N), 10.0, cfg.p0, cfg.profiles, rng)
    noise = before["types"] == NOISE
    np.testing.assert_array_equal(agents.types[noise], NOISE)  # noise never switches
    assert not changed[noise].any()
    assert (agents.types[~noise] == CHART).mean() > 0.95  # strong beta follows the winner
    for k in ("L", "h", "k"):  # fixed
        np.testing.assert_array_equal(getattr(agents, k), before[k])
    same = ~changed
    for k in ("g1", "g2", "n"):  # unchanged agents keep their weights
        np.testing.assert_array_equal(getattr(agents, k)[same], before[k][same])
    lo, hi = cfg.profiles.chartist.g2
    new_c = changed & (agents.types == CHART)
    assert new_c.any() and ((agents.g2[new_c] >= lo) & (agents.g2[new_c] <= hi)).all()
    assert pi[1] > pi[0]


def test_switching_draws_do_not_depend_on_outcome(small_stable: ModelConfig) -> None:
    cfg = small_stable
    states = []
    for winner in (FUND, CHART):
        agents = init_agents(cfg, make_streams(5).init)
        rng = np.random.default_rng(1)
        w = np.where(agents.types == winner, 50.0, 0.0)
        switching_round(agents, w, np.zeros(cfg.N), 10.0, cfg.p0, cfg.profiles, rng)
        states.append(rng.bit_generator.state)
    assert states[0] == states[1]


def test_beta_grid_key_enables_switching(small_stable: ModelConfig) -> None:
    c = with_overrides(small_stable, {"beta": 2.0})
    assert c.switching.enabled and c.switching.beta == 2.0


def test_run_with_switching_records(sw_cfg: ModelConfig) -> None:
    r = run(sw_cfg, 3)
    rounds = sw_cfg.T // sw_cfg.switching.S
    assert r.type_shares.shape == (rounds, 3) and r.switch_pi.shape == (rounds, 2)
    np.testing.assert_allclose(r.type_shares.sum(axis=1), 1.0)
    assert np.ptp(r.type_shares[:, NOISE]) == 0  # noise share never changes
    s = r.summary()
    assert 0 <= s["share_mean_chartist"] <= 1 and "share_sd_chartist" in s
    r2 = run(sw_cfg, 3)
    np.testing.assert_array_equal(r.type_shares, r2.type_shares)
    np.testing.assert_array_equal(r.price, r2.price)


def test_switching_stream_unused_when_off(small_stable: ModelConfig) -> None:
    sim = Simulation(small_stable, 4)
    for t in range(small_stable.T):
        sim.step(t)
    fresh = make_streams(4).switching
    assert sim.streams.switching.bit_generator.state == fresh.bit_generator.state
    assert "share_mean_chartist" not in run(small_stable, 4).summary()


def test_experiment_saves_type_shares(tmp_path: Path) -> None:
    from marketabm.experiments import load_type_shares, run_experiment

    root = Path(__file__).resolve().parents[1]
    exp = ExperimentConfig.model_validate({
        "name": "tiny_sw", "base": "configs/baseline.yaml", "seeds": {"start": 1, "count": 2},
        "grid": {"tau": [0.0, 0.006], "beta": [2.0]},
        "overrides": {"T": 1200, "T_burn": 200, "N": 100, "switching": {"S": 200}},
    })  # fmt: skip
    runs = run_experiment(exp, n_jobs=1, root=root, out_root=tmp_path)
    assert "beta" in runs and "share_mean_chartist" in runs
    ts = load_type_shares("tiny_sw", tmp_path)
    assert set(ts["run_id"]) == set(runs["run_id"])
    assert len(ts) == len(runs) * (1200 // 200)
    assert isinstance(ts, pd.DataFrame)
