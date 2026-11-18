import dataclasses
from pathlib import Path

import numpy as np

from marketabm.agents import init_agents
from marketabm.config import ModelConfig, with_overrides
from marketabm.market import draw_activation, draw_noise, fundamental_path, run
from marketabm.rng import make_streams

ARRAYS = [
    f.name
    for f in dataclasses.fields(__import__("marketabm").SimulationResult)
    if f.name not in ("config", "seed", "metadata", "agent_params")
]


def test_same_seed_identical(small_stable: ModelConfig) -> None:
    a, b = run(small_stable, 21), run(small_stable, 21)
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_different_seed_differs(small_stable: ModelConfig) -> None:
    assert not np.array_equal(run(small_stable, 1).price, run(small_stable, 2).price)


def test_crn_across_tau(small_stable: ModelConfig) -> None:
    c0 = small_stable
    c1 = with_overrides(small_stable, {"tau": 0.006})
    r0, r1 = run(c0, 33), run(c1, 33)
    np.testing.assert_array_equal(r0.fundamental, r1.fundamental)
    for k in r0.agent_params:
        np.testing.assert_array_equal(r0.agent_params[k], r1.agent_params[k])
    np.testing.assert_array_equal(r0.active_by_type, r1.active_by_type)
    # Pre-drawn arrays depend only on the seed, never on tau.
    for cfg in (c0, c1):
        s = make_streams(33)
        fundamental_path(s.fundamental, cfg.T, cfg.p0, cfg.sigma_f)
        init_agents(cfg, s.init)
        act = draw_activation(s.activation, cfg.T, cfg.M, cfg.N)
        noise = draw_noise(s.noise, cfg.T, cfg.M, cfg.sigma_eps)
        if cfg is c0:
            act0, noise0 = act, noise
    np.testing.assert_array_equal(act0, act)
    np.testing.assert_array_equal(noise0, noise)
    assert not np.array_equal(r0.price, r1.price)


def test_run_experiment_parallel_identical(tmp_path) -> None:
    import pandas as pd

    from marketabm.config import ExperimentConfig
    from marketabm.experiments import run_experiment

    root = Path(__file__).resolve().parents[1]
    exp = ExperimentConfig.model_validate({
        "name": "tiny", "base": "configs/baseline.yaml", "seeds": {"start": 1000, "count": 2},
        "grid": {"tau": [0.0, 0.006], "chartist_share": [0.1, 0.4]},
        "overrides": {"T": 1200, "T_burn": 200, "N": 100},
    })  # fmt: skip
    a = run_experiment(exp, n_jobs=1, root=root, out_root=tmp_path / "a")
    b = run_experiment(exp, n_jobs=4, root=root, out_root=tmp_path / "b")
    assert len(a) == 8
    pd.testing.assert_frame_equal(a, b)
    ra = pd.read_parquet(tmp_path / "a" / "tiny" / "runs.parquet")
    rb = pd.read_parquet(tmp_path / "b" / "tiny" / "runs.parquet")
    pd.testing.assert_frame_equal(ra, rb)
    assert (tmp_path / "a" / "tiny" / "config_resolved.yaml").exists()
