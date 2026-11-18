"""Experiment execution: grid/Sobol expansion, parallel runs, result tables (Sections 10, 13).

Outputs in ``results/<name>/``:
- ``runs.parquet``: one row per run with every grid key, the seed and every summary metric.
- ``returns.parquet``: the period returns of each run (needed for ``extreme_moves``).
- ``config_resolved.yaml``: the expanded experiment and run metadata.
- ``series/``: period-end price and fundamental per run, only if ``save_series`` is true.
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
from tqdm.auto import tqdm

from marketabm.agents import TYPE_NAMES
from marketabm.config import (
    ExperimentConfig,
    ModelConfig,
    config_hash,
    resolve_base,
    with_overrides,
)
from marketabm.market import run, run_metadata

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- design
def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid, in the order the keys are listed."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*grid.values())]


def grid_tasks(exp: ExperimentConfig) -> list[dict[str, Any]]:
    """One task per (cell, seed): ``{"cell": {...}, "seed": s}``. Seeds are shared by all cells."""
    assert exp.seeds is not None
    return [
        {"cell": cell, "seed": s} for cell in expand_grid(exp.grid) for s in exp.seeds.as_list()
    ]


def sobol_sample(exp: ExperimentConfig) -> tuple[dict[str, Any], np.ndarray]:
    """SALib problem and sample matrix for E4 (Section 13.3)."""
    assert exp.sobol is not None
    sb = exp.sobol
    problem = {
        "num_vars": len(sb.params),
        "names": list(sb.params),
        "bounds": [list(b) for b in sb.params.values()],
    }
    try:
        from SALib.sample import sobol as sampler

        x = sampler.sample(problem, sb.base_n, calc_second_order=sb.calc_second_order,
                           seed=sb.sample_seed)  # fmt: skip
    except ImportError:  # pragma: no cover - older SALib
        from SALib.sample import saltelli as sampler

        x = sampler.sample(problem, sb.base_n, calc_second_order=sb.calc_second_order)
    return problem, x


def sobol_tasks(exp: ExperimentConfig) -> list[dict[str, Any]]:
    """Two tasks per sample point (old and new tax), seed = seed_start + point index."""
    assert exp.sobol is not None
    sb = exp.sobol
    _, x = sobol_sample(exp)
    tasks = []
    for idx, row in enumerate(x):
        point: dict[str, Any] = {}
        for name, val in zip(sb.params, row, strict=True):
            point[name] = int(round(val)) if name in sb.integer_params else float(val)
        for tau in sb.taus:
            tasks.append({"cell": {**point, "tau": tau}, "seed": sb.seed_start + idx,
                          "point": idx})  # fmt: skip
    return tasks


def cell_config(base: ModelConfig, cell: dict[str, Any], save_series: bool) -> ModelConfig:
    return with_overrides(base, {**cell, "save_series": save_series})


# --------------------------------------------------------------------------- execution
def run_task(base: ModelConfig, task: dict[str, Any], save_series: bool) -> dict[str, Any]:
    """Run one task; returns the summary row plus period returns (and series if asked)."""
    cfg = cell_config(base, task["cell"], save_series)
    res = run(cfg, task["seed"])
    row: dict[str, Any] = {**task["cell"], "seed": task["seed"]}
    if "point" in task:
        row["point"] = task["point"]
    row["config_hash"] = config_hash(cfg)
    row.update(res.summary())
    out: dict[str, Any] = {"row": row, "R": res.period_returns()}
    out["runtime"] = res.metadata["runtime_s"]
    if len(res.switch_t):
        out["type_shares"] = pd.DataFrame(res.type_shares, columns=list(TYPE_NAMES)).assign(
            t=res.switch_t
        )
    if save_series:
        ends = np.arange(cfg.return_interval - 1, cfg.T, cfg.return_interval)
        out["series"] = pd.DataFrame(
            {"t": ends, "price": res.price[ends], "fundamental": res.fundamental[ends]}
        )
    return out


def execute(
    base: ModelConfig,
    tasks: list[dict[str, Any]],
    n_jobs: int = -1,
    save_series: bool = False,
    desc: str = "runs",
) -> list[dict[str, Any]]:
    """Run tasks in parallel (loky); output order equals task order for any ``n_jobs``."""
    par = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")
    jobs = (delayed(run_task)(base, t, save_series) for t in tasks)
    return list(tqdm(par(jobs), total=len(tasks), desc=desc))


def run_experiment(
    exp: ExperimentConfig,
    n_jobs: int = -1,
    root: str | Path = ".",
    out_root: str | Path = "results",
) -> pd.DataFrame:
    """Run a whole experiment and write its result files. Returns the runs table."""
    base = resolve_base(exp, root)
    tasks = sobol_tasks(exp) if exp.sobol is not None else grid_tasks(exp)
    out_dir = Path(out_root) / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = execute(base, tasks, n_jobs, exp.save_series, desc=exp.name)

    runs = pd.DataFrame([r["row"] for r in results])
    runs.insert(0, "run_id", np.arange(len(runs)))
    runs.to_parquet(out_dir / "runs.parquet", index=False)
    pd.DataFrame({"run_id": runs["run_id"], "R": [r["R"].tolist() for r in results]}).to_parquet(
        out_dir / "returns.parquet", index=False
    )
    if exp.save_series:
        _save_series(out_dir / "series", results)
    if any("type_shares" in r for r in results):
        _save_type_shares(out_dir / "type_shares.parquet", runs, results)
    _write_resolved(out_dir, exp, base, len(tasks), sum(r["runtime"] for r in results))
    return runs


def _save_series(directory: Path, results: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        r["series"].to_parquet(directory / f"{i:05d}.parquet", index=False)


def _save_type_shares(path: Path, runs: pd.DataFrame, results: list[dict[str, Any]]) -> None:
    """Type shares after every switching round, long format: run_id, t, one column per type."""
    frames = [
        r["type_shares"].assign(run_id=rid)
        for rid, r in zip(runs["run_id"], results, strict=True)
        if "type_shares" in r
    ]
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def load_type_shares(name: str, out_root: str | Path = "results") -> pd.DataFrame:
    return pd.read_parquet(Path(out_root) / name / "type_shares.parquet")


def _write_resolved(
    out_dir: Path, exp: ExperimentConfig, base: ModelConfig, n_runs: int, cpu_s: float
) -> None:
    meta = run_metadata(base, seed=-1)
    doc = {
        "experiment": exp.model_dump(mode="json"),
        "base_config": base.canonical_dict(),
        "base_config_hash": meta["config_hash"],
        "n_runs": n_runs,
        "total_run_cpu_seconds": round(cpu_s, 1),
        "git_commit": meta["git_commit"],
        "python": meta["python"],
        "versions": meta["versions"],
    }
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)


def load_runs(name: str, out_root: str | Path = "results") -> pd.DataFrame:
    return pd.read_parquet(Path(out_root) / name / "runs.parquet")


def load_returns(name: str, out_root: str | Path = "results") -> dict[int, np.ndarray]:
    df = pd.read_parquet(Path(out_root) / name / "returns.parquet")
    return {int(i): np.asarray(r, dtype=float) for i, r in zip(df["run_id"], df["R"], strict=True)}
