"""E1 fallback: choose a best-compromise calibration point.

No point passes all five validation checks (see results/e1_calibration/pilot_cycle_map.csv):
clustering and |acf_r_lag1| < 0.1 conflict. We take a compromise point, confirm
it at 30 seeds, and document the partial failure.

Usage: uv run python scripts/e1_compromise.py [--n-jobs N]
Writes results/e1_calibration/compromise_runs.parquet, compromise_report.md and
configs/calibrated.yaml.
"""

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
from marketabm.stylized_facts import MOMENT_NAMES

# Anchor: v1.2 point 139 (withdrawn because it cycles); only tau_f and price_band move.
ANCHOR = {
    "sigma_eps": 0.0002,
    "k_max": 0.01,
    "chartist_g2": [0.8, 1.2],
    "vol_feedback_gamma": 3,
    "order_life": 200,
}
TAU_F = [12.5, 15.0, 17.5]
BANDS = [0.0075, 0.00875]
# ASSUMPTION: a compromise point must pass kurtosis, Hill and no-cycle; among those, the lowest
# D wins. D already weighs the acf1 miss against the clustering moments.
REQUIRED = ["kurt>1", "2<hill<6", "|acfR 2-5|<0.1"]
CHECKS = ["kurt>1", "|acf1|<0.1", "acf|R|>band 1-20", "2<hill<6", "|acfR 2-5|<0.1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    cfg = cal.load_calibration_config("configs/e1_calibration.yaml")
    out = Path("results") / cfg.name
    points = {
        i: {**ANCHOR, "tau_f": tf, "price_band": b}
        for i, (tf, b) in enumerate((tf, b) for tf in TAU_F for b in BANDS)
    }
    base = load_model_config(cfg.base)
    runs = cal.run_points(base, points, cfg.seeds_stage2, args.n_jobs, desc="E1 compromise")
    runs.to_parquet(out / "compromise_runs.parquet")

    r = empirical.load_returns()
    m_emp = empirical.moment_vector(r)
    rng = np.random.default_rng(cfg.bootstrap.seed)
    w = empirical.moment_weights(
        empirical.block_bootstrap_moments(r, rng, cfg.bootstrap.block, cfg.bootstrap.n_boot)
    )
    s = cal.score(cal.point_means(runs), m_emp, w).join(cal.points_table(runs))
    s["n_lags_clustered"] = [
        int(sum(row[c] > row["acf_band"] for c in cal.ACF_COLS)) for _, row in s.iterrows()
    ]
    ok = s[s[REQUIRED].all(axis=1) & s["D"].notna()]
    if ok.empty:
        print("No candidate passes kurtosis, Hill and no-cycle.")
        return 2
    chosen = int(ok["D"].idxmin())
    point = points[chosen]
    with open("configs/calibrated.yaml", "w", encoding="utf-8") as fh:
        fh.write(
            f"# E1 best compromise (partial calibration failure): "
            f"{cal.point_label(point)}\n"
        )
        yaml.safe_dump(cal.calibrated_config(cfg.base, point), fh, sort_keys=False)

    cols = ["p_tau_f", "p_price_band", "D", *MOMENT_NAMES, "n_lags_clustered", *CHECKS]
    row = s.loc[chosen]
    cmp = pd.DataFrame(
        {"model (30-seed mean)": [row[m] for m in MOMENT_NAMES], "EPHE": m_emp},
        index=MOMENT_NAMES,
    )
    lines = [
        "# E1 compromise report",
        "",
        "No grid or pilot point passes all five validation checks: volatility clustering and",
        "|acf_r_lag1| < 0.1 conflict (pilot_cycle_map.csv). Measuring returns from the mid price",
        "does not remove the lag-1 reversal, so it is genuine overshoot, not bid-ask bounce.",
        "",
        f"Candidates: anchor {cal.point_label(ANCHOR)}; tau_f in {TAU_F} x price_band in {BANDS};",
        f"{len(cfg.seeds_stage2)} seeds each. Rule: pass {REQUIRED}, then lowest D.",
        "",
        "## Candidates",
        "",
        s[cols].to_string(),
        "",
        f"## Chosen: {cal.point_label(point)}  (D = {row['D']:.4g})",
        "",
        cmp.to_string(),
        "",
        f"Checks: {dict(row[CHECKS])}; |R| ACF above band at {row['n_lags_clustered']}/20 lags.",
        "Saved as configs/calibrated.yaml. This is a partial calibration failure (12.4 step 8).",
    ]
    (out / "compromise_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
