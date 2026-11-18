"""Make the figures of an experiment (CLAUDE.md Section 14.1). Run analyze.py first.

Usage: uv run python scripts/make_figures.py --experiment e2_tax_mix
  e1_calibration  -> F1, F2 (uses configs/calibrated.yaml and the empirical returns)
  e2_tax_mix      -> F3, F4, F5
  e2b_thin_market -> F6
  e3_adaptive     -> F7
  e4_sensitivity  -> F8
Writes PNG (200 dpi) and PDF to results/<experiment>/figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from marketabm import empirical, viz  # noqa: E402
from marketabm.config import load_model_config  # noqa: E402
from marketabm.experiments import load_runs, load_type_shares  # noqa: E402
from marketabm.market import run  # noqa: E402

F3_METRICS = ["vol", "mispricing", "rel_spread_mean", "volume_total", "tax_revenue"]
F4_METRICS = ["vol", "mispricing", "rel_spread_mean", "depth_mean", "volume_total", "tax_revenue"]
FIG_SEED = 1000  # the first experiment seed


def e1(out: Path) -> list[Path]:
    cfg_path = Path("configs/calibrated.yaml")
    if not cfg_path.exists():
        print("configs/calibrated.yaml missing; F1/F2 use configs/baseline.yaml (uncalibrated).")
        cfg_path = Path("configs/baseline.yaml")
    r = run(load_model_config(cfg_path), FIG_SEED)
    paths = viz.save_figure(viz.fig_price_path(r), out, "F1_price_path")
    src = str(empirical.load_source().get("source", "empirical"))
    fig = viz.fig_stylized_facts(r.period_returns(), empirical.load_returns(), src)
    return paths + viz.save_figure(fig, out, "F2_stylized_facts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()
    name = args.experiment
    base = Path("results") / name
    out = base / "figures"
    ana = base / "analysis"
    if name == "e1_calibration":
        paths = e1(out)
    elif name == "e2_tax_mix":
        cm = pd.read_csv(ana / "cell_means.csv")
        ph = pd.read_csv(ana / "ph_cut_effect.csv")
        paths = viz.save_figure(viz.fig_heatmaps(cm, F3_METRICS), out, "F3_heatmaps")
        paths += viz.save_figure(viz.fig_ph_cut(ph, F4_METRICS), out, "F4_ph_cut_effect")
        paths += viz.save_figure(viz.fig_skip_rates(cm), out, "F5_skip_rates")
    elif name == "e2b_thin_market":
        cm = pd.read_csv(ana / "cell_means.csv")
        paths = viz.save_figure(viz.fig_thin_market(cm), out, "F6_thin_market")
    elif name == "e3_adaptive":
        runs = load_runs(name)
        base_cfg = load_model_config(Path("configs/calibrated.yaml"))
        fig = viz.fig_switching(load_type_shares(name), runs, base_cfg.T_burn)
        paths = viz.save_figure(fig, out, "F7_switching")
    elif name == "e4_sensitivity":
        idx = pd.read_csv(ana / "sobol_indices.csv")
        paths = viz.save_figure(viz.fig_sobol(idx), out, "F8_sobol")
    else:
        print(f"No figures defined for {name!r}.")
        return 1
    for p in paths:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
