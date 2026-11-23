"""E1 calibration: grid search on stylized-fact moments.

Stage 1 runs every grid point for a few seeds and stores the simulated moments per run. The
moment distance needs the empirical moments, so it is computed afterwards from the stored runs.
Stage 2 re-runs the best points with more seeds and applies the validation checks.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from marketabm import metrics
from marketabm.config import ModelConfig, load_model_config, with_overrides
from marketabm.market import run
from marketabm.stylized_facts import MOMENT_NAMES, moments, standardize

log = logging.getLogger(__name__)

N_ACF_CHECK = 20  # lags of the |R| ACF checked for clustering
ACF_COLS = [f"acf_abs_{k}" for k in range(1, N_ACF_CHECK + 1)]
CYCLE_LAGS = (2, 3, 4, 5)  # lags of the R ACF checked for cycles (v1.2)
ACF_R_COLS = [f"acf_r_{k}" for k in CYCLE_LAGS]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Bootstrap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block: int = Field(default=20, gt=0)
    n_boot: int = Field(default=1000, gt=0)
    seed: int = 0


class CalibrationConfig(BaseModel):
    """E1 settings (``configs/e1_calibration.yaml``).

    ``grid`` keys are raw config keys, except ``chartist_g2``, which sets
    ``profiles.chartist.g2``, and ``vol_feedback_gamma``, which sets ``vol_feedback.gamma``
    (0 disables the feedback).
    """

    model_config = ConfigDict(extra="forbid")
    name: str = "e1_calibration"
    base: str = "configs/baseline.yaml"
    grid: dict[str, list[Any]]
    seeds_stage1: list[int]
    seeds_stage2: list[int]
    top_k: int = Field(default=10, gt=0)
    bootstrap: Bootstrap = Bootstrap()


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    with open(path, encoding="utf-8") as fh:
        return CalibrationConfig.model_validate(yaml.safe_load(fh))


def grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid as a list of flat point dicts."""
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]


def point_overrides(point: dict[str, Any]) -> dict[str, Any]:
    """Translate a grid point into config overrides."""
    out = dict(point)
    if "chartist_g2" in out:
        out["profiles"] = {"chartist": {"g2": list(out.pop("chartist_g2"))}}
    if "vol_feedback_gamma" in out:
        gamma = float(out.pop("vol_feedback_gamma"))
        out["vol_feedback"] = {"enabled": gamma > 0, "gamma": gamma}
    return out


def point_label(point: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in point.items())


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def return_stats(r: np.ndarray) -> dict[str, float]:
    """Moments, |R| ACF at lags 1-20 and the 95% band for one return series."""
    out: dict[str, float] = dict(moments(r))
    z = standardize(r)
    acf_abs = metrics.acf(np.abs(z), 50)
    for k, col in enumerate(ACF_COLS, start=1):
        out[col] = float(acf_abs[k])
    acf_r = metrics.acf(z, max(CYCLE_LAGS))
    for k, col in zip(CYCLE_LAGS, ACF_R_COLS, strict=True):
        out[col] = float(acf_r[k])
    out["acf_band"] = metrics.acf_band(len(z))
    out["n_returns"] = float(len(z))
    return out


def simulate_point(cfg: ModelConfig, point: dict[str, Any], seed: int) -> dict[str, Any]:
    """Run one (grid point, seed) and return its return statistics. Never raises."""
    row: dict[str, Any] = {"point_id": None, "seed": seed}
    try:
        res = run(with_overrides(cfg, point_overrides(point)), seed)
        r = res.period_returns()
        row.update(return_stats(r))
        row["mispricing"] = metrics.mispricing(
            res.price[cfg.T_burn :], res.fundamental[cfg.T_burn :]
        )
        row["ok"] = bool(np.isfinite(r).all())
    except Exception as exc:  # noqa: BLE001
        log.warning("run failed (%s, seed %s): %s", point_label(point), seed, exc)
        row["ok"] = False
    return row


def run_points(
    cfg: ModelConfig,
    points: dict[int, dict[str, Any]],
    seeds: list[int],
    n_jobs: int = -1,
    desc: str = "E1",
) -> pd.DataFrame:
    """Run every (point_id, seed) pair in parallel; one row per run."""
    tasks = [(pid, p, s) for pid, p in points.items() for s in seeds]
    jobs = (delayed(simulate_point)(cfg, p, s) for _, p, s in tasks)
    par = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")
    rows = list(tqdm(par(jobs), total=len(tasks), desc=desc))
    for (pid, p, _), row in zip(tasks, rows, strict=True):
        row["point_id"] = pid
        row.update({f"p_{k}": str(v) if isinstance(v, list) else v for k, v in p.items()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distance, ranking and checks
# ---------------------------------------------------------------------------


def distance(m_sim: np.ndarray, m_emp: np.ndarray, w: np.ndarray) -> float:
    """``D = sum_i w_i (m_sim_i - m_emp_i)^2``; NaN if any moment is missing."""
    d = np.asarray(m_sim, float) - np.asarray(m_emp, float)
    return float(np.sum(np.asarray(w, float) * d**2))


def point_means(runs: pd.DataFrame) -> pd.DataFrame:
    """Average moments and checks inputs per point (failed runs give NaN means)."""
    cols = [*MOMENT_NAMES, *ACF_COLS, *ACF_R_COLS, "acf_band", "mispricing"]
    g = runs.groupby("point_id")
    out = g[cols].mean()  # NaN-skipping mean
    out["n_ok"] = g["ok"].sum()
    out["n_runs"] = g["ok"].size()
    # A point with any failed run is unusable.
    bad = out["n_ok"] < out["n_runs"]
    out.loc[bad, list(MOMENT_NAMES)] = np.nan
    return out


def checks_from_means(row: pd.Series) -> dict[str, bool]:
    """Validation targets on seed-averaged statistics.

    ASSUMPTION: "averaged" means each statistic (and each |R| ACF lag) is averaged over seeds
    first, then compared with the target; the ACF band uses the mean per-run band.
    The fifth check (no cycles: |ACF of R| < 0.1 at lags 2-5) was added in v1.2 after the
    calibrated point 139 turned out to cycle; data without those lags fails it.
    """
    acf_ok = all(row[c] > row["acf_band"] for c in ACF_COLS)
    no_cycle = all(abs(row[c]) < 0.1 for c in ACF_R_COLS) if ACF_R_COLS[0] in row else False
    return {
        "kurt>1": bool(row["excess_kurtosis"] > 1),
        "|acf1|<0.1": bool(abs(row["acf_r_lag1"]) < 0.1),
        "acf|R|>band 1-20": bool(acf_ok),
        "2<hill<6": bool(2 < row["hill_alpha"] < 6),
        "|acfR 2-5|<0.1": bool(no_cycle),
    }


def score(means: pd.DataFrame, m_emp: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """Add distance ``D`` and the pass/fail checks to a per-point table, sorted by D."""
    out = means.copy()
    sim = out[list(MOMENT_NAMES)].to_numpy()
    out["D"] = [distance(s, m_emp, w) if np.isfinite(s).all() else np.nan for s in sim]
    chk = pd.DataFrame([checks_from_means(r) for _, r in out.iterrows()], index=out.index)
    out = out.join(chk)
    out["n_pass"] = chk.sum(axis=1)
    out["passes"] = out["n_pass"] == chk.shape[1]
    return out.sort_values("D", na_position="last")


def points_table(runs: pd.DataFrame) -> pd.DataFrame:
    """The grid parameters of each point_id (columns prefixed ``p_``)."""
    pcols = [c for c in runs.columns if c.startswith("p_")]
    return runs.groupby("point_id")[pcols].first()


def choose(stage2: pd.DataFrame) -> int | None:
    """Lowest-D point among those that pass every check, or None."""
    passing = stage2[stage2["passes"] & stage2["D"].notna()]
    return None if passing.empty else int(passing["D"].idxmin())


def calibrated_config(base_path: str | Path, point: dict[str, Any]) -> dict[str, Any]:
    """Baseline config dict with the chosen point applied (for ``configs/calibrated.yaml``)."""
    cfg = with_overrides(load_model_config(base_path), point_overrides(point))
    return cfg.canonical_dict()
