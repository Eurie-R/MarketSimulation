"""Run E1 calibration (CLAUDE.md Section 12.4).

Usage:
  uv run python scripts/run_calibration.py --stage 1 [--n-jobs N]   # grid x 3 seeds, no data
  uv run python scripts/run_calibration.py --stage 2 [--n-jobs N]   # top-k x 30 seeds (needs PSEi)
  uv run python scripts/run_calibration.py --stage report           # calibrated.yaml + report.md

Outputs go to results/e1_calibration/.
"""

# ASSUMPTION: E1 has its own driver script; scripts/run_experiment.py (Phase 5) covers E2-E4.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from marketabm import calibration as cal
from marketabm import empirical
from marketabm.config import load_model_config
from marketabm.stylized_facts import MOMENT_NAMES, validation_checks

CONFIG = Path("configs/e1_calibration.yaml")


def out_dir(cfg: cal.CalibrationConfig) -> Path:
    d = Path("results") / cfg.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def empirical_targets(cfg: cal.CalibrationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(returns, m_emp, weights). Exits with instructions if the PSEi data is missing."""
    if not empirical.PROCESSED.exists():
        print(
            f"Missing {empirical.PROCESSED}. Run scripts/fetch_psei.py first "
            f"(or place data/raw/psei.csv with columns Date,Close and run it)."
        )
        sys.exit(1)
    r = empirical.load_returns()
    m_emp = empirical.moment_vector(r)
    rng = np.random.default_rng(cfg.bootstrap.seed)
    boot = empirical.block_bootstrap_moments(r, rng, cfg.bootstrap.block, cfg.bootstrap.n_boot)
    return r, m_emp, empirical.moment_weights(boot)


def stage1(cfg: cal.CalibrationConfig, n_jobs: int) -> None:
    base = load_model_config(cfg.base)
    points = dict(enumerate(cal.grid_points(cfg.grid)))
    print(f"Stage 1: {len(points)} points x {len(cfg.seeds_stage1)} seeds")
    runs = cal.run_points(base, points, cfg.seeds_stage1, n_jobs, desc="E1 stage 1")
    path = out_dir(cfg) / "stage1_runs.parquet"
    runs.to_parquet(path)
    print(f"Saved {path}")
    means = cal.point_means(runs)
    chk = pd.DataFrame([cal.checks_from_means(r) for _, r in means.iterrows()])
    print("Points passing each 12.3 check (3-seed means):")
    print(chk.sum().to_string())
    print(f"Points passing all: {int(chk.all(axis=1).sum())}")


def stage2(cfg: cal.CalibrationConfig, n_jobs: int) -> None:
    _, m_emp, w = empirical_targets(cfg)
    d = out_dir(cfg)
    runs1 = pd.read_parquet(d / "stage1_runs.parquet")
    s1 = cal.score(cal.point_means(runs1), m_emp, w)
    s1.join(cal.points_table(runs1)).to_parquet(d / "stage1_scored.parquet")
    grid = cal.grid_points(cfg.grid)
    # ASSUMPTION (deviation from 12.4 step 6, made after the v1.2 run): stage 2 re-runs the
    # top_k points by D *and* every point whose 3-seed means pass all 12.3 checks. With 3 seeds
    # D is too noisy to rank on: in v1.2 none of the top 10 passed at 30 seeds, while 4 of the 8
    # stage-1 passers did, and the best of them had a lower 30-seed D than any top-10 point.
    top = [int(i) for i in s1.index[: cfg.top_k]]
    top += [int(i) for i in s1.index[s1["passes"]] if int(i) not in top]
    base = load_model_config(cfg.base)
    runs2 = cal.run_points(
        base, {i: grid[i] for i in top}, cfg.seeds_stage2, n_jobs, desc="E1 stage 2"
    )
    runs2.to_parquet(d / "stage2_runs.parquet")
    print(f"Saved {d / 'stage2_runs.parquet'}")


def _cell(v: object) -> str:
    return f"{v:.4g}" if isinstance(v, float | np.floating) else str(v)


def _fmt_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    head = ["", *map(str, df.columns)]
    rows = [[str(i), *(_cell(v) for v in r)] for i, r in zip(df.index, df.to_numpy(), strict=True)]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def report(cfg: cal.CalibrationConfig) -> int:
    r_emp, m_emp, w = empirical_targets(cfg)
    d = out_dir(cfg)
    runs2 = pd.read_parquet(d / "stage2_runs.parquet")
    s2 = cal.score(cal.point_means(runs2), m_emp, w).join(cal.points_table(runs2))
    s2.to_parquet(d / "stage2_scored.parquet")
    chosen = cal.choose(s2)
    grid = cal.grid_points(cfg.grid)

    pcols = [c for c in s2.columns if c.startswith("p_")]
    show = s2[[*pcols, "D", *MOMENT_NAMES, "n_pass", "passes"]]
    emp_checks = validation_checks(r_emp)
    meta = empirical.load_source()
    src = str(meta["source"])
    lines = [
        "# E1 calibration report",
        "",
        f"Empirical target: {src}, daily log returns {meta.get('start', '?')} to "
        f"{meta.get('end', '?')}, n = {len(r_emp)}. Moments use standardized returns.",
        "",
        "## Empirical moments and bootstrap weights",
        "",
        _fmt_table(pd.DataFrame({src: m_emp, "weight (1/var)": w}, index=MOMENT_NAMES)),
        "",
        f"{src} 12.3 checks: {emp_checks}",
        "",
        f"## Top {len(s2)} points ({len(cfg.seeds_stage2)} seeds each, sorted by D)",
        "",
        _fmt_table(show),
        "",
    ]
    if chosen is None:
        lines += [
            "## Result: no point passes all Section 12.3 checks",
            "",
            "Per-check failures among the top points:",
            "",
            _fmt_table(
                s2[["kurt>1", "|acf1|<0.1", "acf|R|>band 1-20", "2<hill<6", "|acfR 2-5|<0.1"]]
            ),
        ]
        print("\n".join(lines[-6:]))
    else:
        point = grid[chosen]
        cal_cfg = cal.calibrated_config(cfg.base, point)
        with open("configs/calibrated.yaml", "w", encoding="utf-8") as fh:
            fh.write(f"# Calibrated by E1 (point {chosen}: {cal.point_label(point)})\n")
            yaml.safe_dump(cal_cfg, fh, sort_keys=False)
        row = s2.loc[chosen]
        cmp = pd.DataFrame(
            {"model (30-seed mean)": [row[m] for m in MOMENT_NAMES], src: m_emp},
            index=MOMENT_NAMES,
        )
        lines += [
            f"## Chosen point: {chosen}",
            "",
            cal.point_label(point),
            "",
            f"D = {row['D']:.4g}",
            "",
            _fmt_table(cmp),
            "",
            "Saved as `configs/calibrated.yaml`.",
        ]
        print(f"Chosen point {chosen}: {cal.point_label(point)}  (D = {row['D']:.4g})")
    (d / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {d / 'report.md'}")
    return 0 if chosen is not None else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", choices=["1", "2", "report"], required=True)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    cfg = cal.load_calibration_config(args.config)
    if args.stage == "1":
        stage1(cfg, args.n_jobs)
    elif args.stage == "2":
        stage2(cfg, args.n_jobs)
    else:
        return report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
