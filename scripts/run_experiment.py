"""Run an experiment (CLAUDE.md Section 13).

Usage: uv run python scripts/run_experiment.py --config configs/e2_tax_mix.yaml [--n-jobs N]
       [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from marketabm.config import load_experiment_config
from marketabm.experiments import grid_tasks, run_experiment, sobol_tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true", help="list the runs without executing")
    args = ap.parse_args()

    exp = load_experiment_config(args.config)
    if not Path(exp.base).exists():
        print(
            f"WARNING: {exp.base} not found; falling back to configs/baseline.yaml "
            "(uncalibrated). Run E1 first for the real experiment."
        )
    tasks = sobol_tasks(exp) if exp.sobol is not None else grid_tasks(exp)
    print(f"{exp.name}: {len(tasks)} runs")
    if args.dry_run:
        for t in tasks[:10]:
            print(" ", t)
        if len(tasks) > 10:
            print(f"  ... ({len(tasks) - 10} more)")
        return 0
    t0 = time.perf_counter()
    runs = run_experiment(exp, n_jobs=args.n_jobs)
    print(
        f"Wrote results/{exp.name}/runs.parquet ({len(runs)} rows) in "
        f"{(time.perf_counter() - t0) / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
