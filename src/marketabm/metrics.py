"""Metric definitions. Pure functions: arrays in, numbers out."""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf as _sm_acf

log = logging.getLogger(__name__)

NAN = float("nan")


def period_returns(price: np.ndarray, T_burn: int, interval: int) -> np.ndarray:  # noqa: N803
    """Log changes between consecutive period-end prices after burn-in (Section 5.7).

    ``R_j = log P[T_burn + (j+1)*interval - 1] - log P[T_burn + j*interval - 1]``.
    For ``T_burn = 0`` the first base price is index ``interval - 1``.
    """
    logp = np.log(np.asarray(price, dtype=float))
    start = T_burn - 1 if T_burn >= 1 else interval - 1
    ends = logp[start::interval]
    return np.diff(ends)


def acf(x: np.ndarray, nlags: int = 50) -> np.ndarray:
    """Sample ACF (lags 0..nlags) via statsmodels with FFT; NaN if undefined."""
    x = np.asarray(x, dtype=float)
    if len(x) <= nlags + 1 or not np.isfinite(x).all() or np.std(x) == 0:
        return np.full(nlags + 1, NAN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(_sm_acf(x, nlags=nlags, fft=True))


def acf_band(n: int) -> float:
    """Half-width of the 95% band for an i.i.d. ACF, ``1.96 / sqrt(n)``."""
    return 1.96 / math.sqrt(n) if n > 0 else NAN


def vol(r: np.ndarray) -> float:
    """Sample standard deviation (ddof=1)."""
    return float(np.std(r, ddof=1)) if len(r) >= 2 else NAN


def acf_summary(r: np.ndarray) -> dict[str, float]:
    """``acf_r_lag1``, ``acf_absr_1_10`` and ``acf_absr_11_50``."""
    a = acf(r, 50)
    b = acf(np.abs(r), 50)
    return {
        "acf_r_lag1": float(a[1]),
        "acf_absr_1_10": float(np.mean(b[1:11])),
        "acf_absr_11_50": float(np.mean(b[11:51])),
    }


def excess_kurtosis(r: np.ndarray) -> float:
    if len(r) < 4 or np.std(r) == 0:
        return NAN
    return float(stats.kurtosis(r, fisher=True, bias=False))


def hill_alpha(r: np.ndarray, frac: float = 0.05) -> float:
    """Hill tail index on |r| using the top ``frac`` of observations."""
    x = np.sort(np.abs(np.asarray(r, dtype=float)))[::-1]
    k = int(math.floor(frac * len(x)))
    if k < 1 or k >= len(x) or x[k] <= 0:
        return NAN
    mean_log = float(np.mean(np.log(x[:k] / x[k])))
    return 1.0 / mean_log if mean_log > 0 else NAN


def max_drawdown(logp: np.ndarray) -> float:
    """Largest peak-to-trough fall in a log-price series, as a positive number."""
    logp = np.asarray(logp, dtype=float)
    if len(logp) == 0:
        return NAN
    return float(np.max(np.maximum.accumulate(logp) - logp))


def rel_spread_mean(best_bid: np.ndarray, best_ask: np.ndarray) -> float:
    ok = np.isfinite(best_bid) & np.isfinite(best_ask)
    if not ok.any():
        return NAN
    b, a = best_bid[ok], best_ask[ok]
    return float(np.mean((a - b) / ((a + b) / 2.0)))


def one_sided_share(best_bid: np.ndarray, best_ask: np.ndarray) -> float:
    if len(best_bid) == 0:
        return NAN
    return float(np.mean(~(np.isfinite(best_bid) & np.isfinite(best_ask))))


def depth_mean(depth_bid: np.ndarray, depth_ask: np.ndarray) -> float:
    d = depth_bid + depth_ask
    ok = np.isfinite(d)
    return float(np.mean(d[ok])) if ok.any() else NAN


def mispricing(price: np.ndarray, fundamental: np.ndarray) -> float:
    if len(price) == 0:
        return NAN
    return float(np.mean(np.abs(np.log(price) - np.log(fundamental))))


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else NAN


def composition(
    fills: np.ndarray,
    aggressive: np.ndarray,
    skipped_tax: np.ndarray,
    skipped_no_signal: np.ndarray,
    active: np.ndarray,
    agent_counts: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, float]:
    """Section 11.5 metrics from per-type totals (each shape (3,))."""
    total = fills.sum()
    out: dict[str, float] = {}
    for k, name in enumerate(names):
        out[f"fills_per_agent_{name}"] = safe_div(fills[k], agent_counts[k])
        out[f"volume_share_{name}"] = safe_div(fills[k], total)
        out[f"aggressive_share_{name}"] = safe_div(aggressive[k], fills[k])
        out[f"skip_rate_tax_{name}"] = safe_div(skipped_tax[k], active[k])
        out[f"skip_rate_no_signal_{name}"] = safe_div(skipped_no_signal[k], active[k])
    return out


def return_metrics(r: np.ndarray) -> dict[str, float]:
    """Volatility, clustering and tail metrics on period returns."""
    out = {"vol": vol(r), "excess_kurtosis": excess_kurtosis(r), "hill_alpha": hill_alpha(r)}
    out.update(acf_summary(r))
    return out
