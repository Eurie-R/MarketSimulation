"""Configuration models, YAML loading and mix derivation (CLAUDE.md Section 9)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TYPE_KEYS: tuple[str, str, str] = ("fundamentalist", "chartist", "noise")
MIX_TOL = 1e-9


def _check_range(v: list[float] | tuple[float, float], name: str) -> tuple[float, float]:
    if len(v) != 2:
        raise ValueError(f"{name} must be a [low, high] pair, got {v!r}")
    low, high = v
    if low > high:
        raise ValueError(f"{name} must have low <= high, got [{low}, {high}]")
    return (low, high)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Mix(_Strict):
    """Population shares by trader type; must sum to 1."""

    fundamentalist: float = Field(ge=0.0, le=1.0)
    chartist: float = Field(ge=0.0, le=1.0)
    noise: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> Mix:
        total = self.fundamentalist + self.chartist + self.noise
        if abs(total - 1.0) > MIX_TOL:
            raise ValueError(f"mix shares must sum to 1 (within {MIX_TOL}), got {total!r}")
        return self

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.fundamentalist, self.chartist, self.noise)


class Profile(_Strict):
    """Parameter ranges for one trader type."""

    g1: tuple[float, float]
    g2: tuple[float, float]
    n: tuple[float, float]

    @field_validator("g1", "g2", "n", mode="before")
    @classmethod
    def _range(cls, v: Any, info: Any) -> tuple[float, float]:
        return _check_range(list(v), info.field_name)

    @model_validator(mode="after")
    def _non_negative(self) -> Profile:
        for name in ("g1", "g2", "n"):
            if getattr(self, name)[0] < 0:
                raise ValueError(f"profile weight {name} must be >= 0")
        if self.g1[0] + self.g2[0] + self.n[0] <= 0:
            raise ValueError("profile weights g1 + g2 + n must be > 0 for every agent")
        return self


class Profiles(_Strict):
    fundamentalist: Profile
    chartist: Profile
    noise: Profile

    def by_index(self) -> tuple[Profile, Profile, Profile]:
        return (self.fundamentalist, self.chartist, self.noise)


class Switching(_Strict):
    enabled: bool = False
    S: int = Field(default=500, gt=0)
    beta: float | None = None

    @model_validator(mode="after")
    def _beta_required(self) -> Switching:
        # ASSUMPTION: switching cannot run without a beta value, so enabling it with
        # beta = null is rejected rather than silently defaulted.
        if self.enabled and self.beta is None:
            raise ValueError("switching.enabled is true but switching.beta is null")
        return self


class VolFeedback(_Strict):
    """Volatility feedback on the noise term (Section 5.10, v1.2). Off by default."""

    enabled: bool = False
    gamma: float = Field(default=1.0, ge=0)
    fast_halflife: float = Field(default=20.0, gt=0)
    slow_halflife: float = Field(default=2000.0, gt=0)
    cap: float = Field(default=5.0, ge=1)

    @model_validator(mode="after")
    def _halflives(self) -> VolFeedback:
        if self.fast_halflife >= self.slow_halflife:
            raise ValueError(
                f"vol_feedback.fast_halflife ({self.fast_halflife}) must be below "
                f"slow_halflife ({self.slow_halflife})"
            )
        return self


class ModelConfig(_Strict):
    """All model parameters (CLAUDE.md Section 9.2)."""

    # Market
    T: int = Field(default=50_000, gt=0)
    T_burn: int = Field(default=5_000, ge=0)
    M: int = Field(default=10, gt=0)
    N: int = Field(default=1_000, gt=0)
    p0: float = Field(default=100.0, gt=0)
    tick: float = 0.01
    sigma_f: float = Field(default=0.001, ge=0)
    order_life: int = Field(default=200, gt=0)
    return_interval: int = Field(default=20, gt=0)
    # Tax and costs
    tau: float = 0.0
    c_min: float = Field(default=0.0, ge=0)
    # Decision rule
    tau_f: float = Field(default=10.0, gt=0)
    sigma_eps: float = Field(default=0.0005, ge=0)
    k_max: float = Field(default=0.005, ge=0)
    # Order price band and order size (Section 5.5, steps 3-5)
    price_band: float | None = Field(default=0.01, gt=0)
    order_size: Literal["unit", "signal"] = "signal"
    size_unit: float = Field(default=0.01, gt=0)
    size_max: int = Field(default=20, ge=1)
    L_range: tuple[int, int] = (5, 50)
    h_range: tuple[int, int] = (10, 100)
    # Population
    mix: Mix = Mix(fundamentalist=0.45, chartist=0.25, noise=0.30)
    profiles: Profiles = Profiles(
        fundamentalist=Profile(g1=(0.8, 1.2), g2=(0.0, 0.2), n=(0.1, 0.3)),
        chartist=Profile(g1=(0.0, 0.2), g2=(0.8, 1.2), n=(0.1, 0.3)),
        noise=Profile(g1=(0.0, 0.0), g2=(0.0, 0.0), n=(1.0, 1.0)),
    )
    # Volatility feedback (Section 5.10)
    vol_feedback: VolFeedback = VolFeedback()
    # Switching
    switching: Switching = Switching()
    # Output
    save_series: bool = True
    depth_pct: float = Field(default=0.01, gt=0)

    @field_validator("L_range", "h_range", mode="before")
    @classmethod
    def _int_range(cls, v: Any, info: Any) -> tuple[int, int]:
        low, high = _check_range(list(v), info.field_name)
        if low < 1:
            raise ValueError(f"{info.field_name} low must be >= 1, got {low}")
        return (int(low), int(high))

    @field_validator("tau")
    @classmethod
    def _tau(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"tau must be >= 0, got {v}")
        return v

    @field_validator("tick")
    @classmethod
    def _tick(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"tick must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> ModelConfig:
        if self.T_burn >= self.T:
            raise ValueError(f"T ({self.T}) must be greater than T_burn ({self.T_burn})")
        if (self.T - self.T_burn) % self.return_interval != 0:
            raise ValueError(
                f"(T - T_burn) = {self.T - self.T_burn} must be divisible by "
                f"return_interval = {self.return_interval}"
            )
        if self.M > self.N:
            raise ValueError(f"M ({self.M}) must be <= N ({self.N})")
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Plain JSON-compatible dict (tuples become lists)."""
        return json.loads(self.model_dump_json())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def config_hash(cfg: ModelConfig) -> str:
    """SHA-256 of the canonical (sorted-key) JSON of the config."""
    payload = json.dumps(cfg.canonical_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def mix_from_chartist_share(c: float, base_mix: Mix | dict[str, float]) -> dict[str, float]:
    """Mix with chartist share ``c``; the other types keep their baseline ratio (Section 9.3).

    Parameters
    ----------
    c : float
        Target chartist share in [0, 1].
    base_mix : Mix or dict
        Baseline mix providing the fundamentalist : noise ratio.

    Returns
    -------
    dict
        Shares keyed by type name, summing to 1.
    """
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"chartist share must be in [0, 1], got {c}")
    base = base_mix.model_dump() if isinstance(base_mix, Mix) else dict(base_mix)
    rest = base["fundamentalist"] + base["noise"]
    if rest <= 0:
        raise ValueError("base mix has no fundamentalists or noise traders to rescale")
    return {
        "fundamentalist": (1.0 - c) * base["fundamentalist"] / rest,
        "chartist": c,
        "noise": (1.0 - c) * base["noise"] / rest,
    }


def _deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in upd.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = val
    return out


def with_overrides(cfg: ModelConfig, overrides: dict[str, Any]) -> ModelConfig:
    """Return a new validated config with ``overrides`` applied.

    The derived key ``chartist_share`` is converted to a mix via
    :func:`mix_from_chartist_share`; the derived key ``beta`` sets ``switching.beta`` and turns
    switching on; every
    other key must be a raw config key. Nested dicts (e.g. ``switching``) are merged.
    """
    overrides = dict(overrides)
    data = cfg.canonical_dict()
    if "chartist_share" in overrides:
        c = float(overrides.pop("chartist_share"))
        overrides["mix"] = mix_from_chartist_share(c, cfg.mix)
    if "beta" in overrides:
        # ASSUMPTION: E3 varies beta as a flat grid key (like chartist_share), so runs.parquet
        # gets a plain "beta" column. Setting it also turns switching on (a base config with
        # switching on would need a beta of its own to validate).
        sw = dict(overrides.get("switching", {}))
        sw["beta"] = float(overrides.pop("beta"))
        sw["enabled"] = True
        overrides["switching"] = sw
    return ModelConfig.model_validate(_deep_update(data, overrides))


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a model config YAML file."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return ModelConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Experiment config (used from Phase 5 on)
# ---------------------------------------------------------------------------


class Seeds(_Strict):
    start: int = Field(ge=0)
    count: int = Field(gt=0)

    def as_list(self) -> list[int]:
        return list(range(self.start, self.start + self.count))


class Sobol(_Strict):
    """E4 global sensitivity design (Section 13.3)."""

    params: dict[str, tuple[float, float]]
    integer_params: list[str] = Field(default_factory=list)
    base_n: int = Field(default=64, gt=0)
    calc_second_order: bool = True
    seed_start: int = Field(default=5000, ge=0)
    taus: tuple[float, float] = (0.006, 0.001)  # (old, new): Y = metric(new) - metric(old)
    outputs: list[str] = Field(default_factory=lambda: ["vol", "mispricing", "rel_spread_mean"])
    sample_seed: int = 0

    @field_validator("params")
    @classmethod
    def _bounds(cls, v: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        for key, (lo, hi) in v.items():
            if key not in ModelConfig.model_fields:
                raise ValueError(f"unknown Sobol parameter {key!r}")
            if not lo < hi:
                raise ValueError(f"Sobol bounds for {key!r} need low < high, got {lo}, {hi}")
        return v


class ExperimentConfig(_Strict):
    """An experiment: base config + grid + seeds (Section 9.4), or a Sobol design (13.3)."""

    name: str
    base: str
    seeds: Seeds | None = None
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    save_series: bool = False
    sobol: Sobol | None = None

    @model_validator(mode="after")
    def _design(self) -> ExperimentConfig:
        if self.sobol is None and self.seeds is None:
            raise ValueError("a grid experiment needs seeds")
        if self.sobol is not None and self.grid:
            raise ValueError("an experiment has either a grid or a sobol design, not both")
        return self

    @field_validator("grid")
    @classmethod
    def _grid_keys(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        allowed = set(ModelConfig.model_fields) | {"chartist_share", "beta"}
        for key, vals in v.items():
            if key not in allowed:
                raise ValueError(f"unknown grid key {key!r}")
            if not vals:
                raise ValueError(f"grid key {key!r} has no values")
        return v


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config YAML file."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return ExperimentConfig.model_validate(data)


def resolve_base(exp: ExperimentConfig, root: str | Path = ".") -> ModelConfig:
    """Load the experiment's base config, falling back to baseline.yaml if it is missing."""
    root = Path(root)
    path = root / exp.base
    if not path.exists():
        path = root / "configs" / "baseline.yaml"
    return with_overrides(load_model_config(path), exp.overrides)
