"""Analyze an experiment (CLAUDE.md Section 13.4).

Usage: uv run python scripts/analyze.py --experiment e2_tax_mix
Reads results/<name>/ and configs/<name>.yaml; writes results/<name>/analysis/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from marketabm import analysis as an
from marketabm.config import load_experiment_config
from marketabm.experiments import load_returns, load_runs, sobol_sample


def analyze_grid(name: str, exp, out: Path) -> None:
    runs = load_runs(name)
    by = [k for k in exp.grid if k != "tau"]
    runs["extreme_moves"] = an.extreme_moves(runs, load_returns(name), by)
    runs.to_parquet(out / "runs_with_extreme_moves.parquet", index=False)
    metrics = [m for m in an.SUMMARY_METRICS if m in runs] + [
        c
        for c in runs.columns
        if c.startswith(("fills_per_agent_", "skip_rate_tax_", "volume_share_", "share_"))
    ]
    an.cell_means(runs, [*by, "tau"], metrics).to_csv(out / "cell_means.csv", index=False)
    diffs = [an.paired_differences(runs, m, by) for m in metrics]
    pd.concat(diffs, ignore_index=True).to_csv(out / "paired_differences.csv", index=False)
    ph = an.ph_cut_effect(runs, metrics, by)
    ph.to_csv(out / "ph_cut_effect.csv", index=False)

    verdicts = []
    if "chartist_share" in by:
        verdicts += [an.check_h1(runs, by), an.check_h2(runs)]
    if "N" in by:
        verdicts.append(an.check_h3(runs))
    if verdicts:
        (out / "hypotheses.md").write_text(an.hypotheses_markdown(verdicts, name), encoding="utf-8")
        for v in verdicts:
            print(f"{v.hypothesis}: {v.verdict} ({v.n_comparisons} comparisons)")
    key = ph[ph["metric"].isin(["vol", "mispricing", "rel_spread_mean"])]
    print("PH cut effect, metric(0.1%) - metric(0.6%):")
    print(key.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def analyze_sobol(name: str, exp, out: Path) -> None:
    runs = load_runs(name)
    problem, _ = sobol_sample(exp)
    y = an.sobol_outputs(runs, exp.sobol.outputs)
    y.to_csv(out / "sobol_outputs.csv")
    idx = an.sobol_indices(y, problem, exp.sobol.calc_second_order)
    idx.to_csv(out / "sobol_indices.csv", index=False)
    print(idx.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", default=None, help="defaults to configs/<experiment>.yaml")
    args = ap.parse_args()
    exp = load_experiment_config(args.config or f"configs/{args.experiment}.yaml")
    out = Path("results") / args.experiment / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if exp.sobol is not None:
        analyze_sobol(args.experiment, exp, out)
    else:
        analyze_grid(args.experiment, exp, out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
