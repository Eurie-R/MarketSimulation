"""Stylized-fact moments and the baseline validation checks (Sections 12.2-12.3)."""

from __future__ import annotations

import numpy as np

from marketabm import metrics

MOMENT_NAMES: tuple[str, ...] = (
    "excess_kurtosis",
    "hill_alpha",
    "acf_r_lag1",
    "acf_absr_1_10",
    "acf_absr_11_50",
)


def standardize(r: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance (ddof=1)."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return (r - r.mean()) / sd if sd > 0 else r - r.mean()


def moments(r: np.ndarray) -> dict[str, float]:
    """Moment vector of Section 12.2 on standardized returns."""
    z = standardize(r)
    out = {
        "excess_kurtosis": metrics.excess_kurtosis(z),
        "hill_alpha": metrics.hill_alpha(z),
    }
    out.update(metrics.acf_summary(z))
    return {k: out[k] for k in MOMENT_NAMES}


def validation_checks(r: np.ndarray, max_lag: int = 20) -> dict[str, bool]:
    """Pass/fail for the Section 12.3 targets on one return series."""
    z = standardize(r)
    m = moments(z)
    acf_abs = metrics.acf(np.abs(z), 50)
    acf_r = metrics.acf(z, 5)
    band = metrics.acf_band(len(z))
    return {
        "excess_kurtosis > 1": bool(m["excess_kurtosis"] > 1),
        "|acf_r_lag1| < 0.1": bool(abs(m["acf_r_lag1"]) < 0.1),
        f"acf|R| > band, lags 1-{max_lag}": bool(np.all(acf_abs[1 : max_lag + 1] > band)),
        "2 < hill_alpha < 6": bool(2 < m["hill_alpha"] < 6),
        "no cycle: |acf_r| < 0.1 at lags 2-5": bool(np.all(np.abs(acf_r[2:6]) < 0.1)),
    }
