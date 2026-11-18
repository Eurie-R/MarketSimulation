from pathlib import Path

import pytest

from marketabm.config import ModelConfig, load_model_config, with_overrides

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def baseline() -> ModelConfig:
    return load_model_config(ROOT / "configs" / "baseline.yaml")


@pytest.fixture(scope="session")
def small_stable(baseline: ModelConfig) -> ModelConfig:
    """Short run at the baseline mix (chartists included)."""
    return with_overrides(baseline, {"T": 2000, "T_burn": 500, "N": 100})
