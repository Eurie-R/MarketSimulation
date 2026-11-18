import pytest
from pydantic import ValidationError

from marketabm.config import (
    ExperimentConfig,
    ModelConfig,
    config_hash,
    mix_from_chartist_share,
    with_overrides,
)


def test_baseline_matches_defaults(baseline: ModelConfig) -> None:
    assert baseline == ModelConfig()
    assert baseline.T == 50_000 and baseline.N == 1000 and baseline.tau == 0.0
    assert baseline.mix.as_tuple() == (0.45, 0.25, 0.30)
    assert baseline.tau_f == 10 and baseline.price_band == 0.01
    assert baseline.order_size == "signal" and baseline.size_unit == 0.01
    assert baseline.size_max == 20


@pytest.mark.parametrize(
    ("override", "msg"),
    [
        ({"mix": {"fundamentalist": 0.5, "chartist": 0.5, "noise": 0.1}}, "sum to 1"),
        ({"mix": {"fundamentalist": 1.2, "chartist": -0.2, "noise": 0.0}}, "less than or equal"),
        ({"L_range": [50, 5]}, "low <= high"),
        ({"profiles": {"chartist": {"g2": [1.2, 0.8]}}}, "low <= high"),
        ({"T": 5000}, "greater than T_burn"),
        ({"T": 50010}, "divisible"),
        ({"M": 2000}, r"M \(2000\) must be <= N"),
        ({"tau": -0.01}, "tau must be >= 0"),
        ({"tick": 0.0}, "tick must be > 0"),
        ({"switching": {"enabled": True}}, "beta is null"),
        ({"not_a_key": 1}, "Extra inputs"),
        ({"price_band": 0.0}, "greater than 0"),
        ({"order_size": "big"}, "unit"),
        ({"size_max": 0}, "greater than or equal to 1"),
    ],
)
def test_invalid_configs_rejected(baseline: ModelConfig, override: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        with_overrides(baseline, override)


def test_overrides_and_hash(baseline: ModelConfig) -> None:
    cfg = with_overrides(baseline, {"tau": 0.006, "chartist_share": 0.4})
    assert cfg.tau == 0.006
    assert cfg.mix.chartist == 0.4
    assert config_hash(cfg) != config_hash(baseline)
    assert config_hash(with_overrides(baseline, {})) == config_hash(baseline)


def test_mix_from_chartist_share() -> None:
    base = {"fundamentalist": 0.45, "chartist": 0.25, "noise": 0.30}
    for c in (0.0, 0.1, 0.25, 0.4, 0.55, 1.0):
        m = mix_from_chartist_share(c, base)
        assert sum(m.values()) == pytest.approx(1.0, abs=1e-12)
        assert m["fundamentalist"] == pytest.approx(0.6 * (1 - c))
        assert m["noise"] == pytest.approx(0.4 * (1 - c))
        assert m["chartist"] == c
    with pytest.raises(ValueError):
        mix_from_chartist_share(1.5, base)


def test_experiment_config_rejects_unknown_grid_key() -> None:
    with pytest.raises(ValidationError, match="unknown grid key"):
        ExperimentConfig.model_validate(
            {"name": "x", "base": "b.yaml", "seeds": {"start": 1, "count": 2}, "grid": {"zz": [1]}}
        )


def test_vol_feedback_halflives_validated(baseline) -> None:
    import pytest
    from pydantic import ValidationError

    from marketabm.config import with_overrides

    with pytest.raises(ValidationError, match="fast_halflife"):
        with_overrides(baseline, {"vol_feedback": {"fast_halflife": 3000}})


def test_e4_sobol_design() -> None:
    from pathlib import Path

    from marketabm.config import load_experiment_config
    from marketabm.experiments import sobol_tasks

    root = Path(__file__).resolve().parents[1]
    exp = load_experiment_config(root / "configs" / "e4_sensitivity.yaml")
    tasks = sobol_tasks(exp)
    assert len(tasks) == 1024  # 64 x (2*3 + 2) points x 2 tax rates
    assert all(isinstance(t["cell"]["N"], int) for t in tasks)
    assert tasks[0]["seed"] == 5000 and tasks[-1]["seed"] == 5000 + 511
    assert {t["cell"]["tau"] for t in tasks} == {0.006, 0.001}
    assert all(250 <= t["cell"]["N"] <= 2000 for t in tasks)


def test_experiment_needs_seeds_or_sobol() -> None:
    import pytest
    from pydantic import ValidationError

    from marketabm.config import ExperimentConfig

    with pytest.raises(ValidationError, match="needs seeds"):
        ExperimentConfig.model_validate({"name": "x", "base": "b.yaml", "grid": {"tau": [0.0]}})
