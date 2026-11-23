"""Statistical analysis: cell means, paired differences, PH cut effect, hypotheses.

All differences are paired by seed: the same seed in two cells shares every random number,
so ``metric(cell A, seed s) - metric(cell B, seed s)`` isolates the parameter change.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from marketabm.agents import TYPE_NAMES

PH_OLD, PH_NEW = 0.006, 0.001
SUMMARY_METRICS: tuple[str, ...] = (
    "vol", "mispricing", "rel_spread_mean", "depth_mean", "volume_total", "tax_revenue",
    "excess_kurtosis", "hill_alpha", "max_drawdown", "one_sided_share", "book_size_mean",
    "extreme_moves",
)  # fmt: skip


# --------------------------------------------------------------------------- intervals
def t_interval(
    x: Sequence[float] | np.ndarray, level: float = 0.95
) -> tuple[float, float, float, int]:
    """Mean and two-sided t-interval of the finite values of ``x``: (mean, low, high, n)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return (math.nan, math.nan, math.nan, 0)
    m = float(a.mean())
    if n < 2:
        return (m, math.nan, math.nan, n)
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * a.std(ddof=1) / math.sqrt(n))
    return (m, m - half, m + half, n)


def cell_means(runs: pd.DataFrame, by: list[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean and 95% t-interval across seeds for every metric in every cell (long format)."""
    rows = []
    for key, g in runs.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            if metric in g:
                rows.append((*key, metric, *t_interval(g[metric].to_numpy())))
    return pd.DataFrame(rows, columns=[*by, "metric", "mean", "ci_low", "ci_high", "n"])


# --------------------------------------------------------------------------- pairing
def paired_values(runs: pd.DataFrame, metric: str, by: list[str], key: str = "tau") -> pd.DataFrame:
    """Table indexed by (by..., seed) with one column per value of ``key``."""
    return runs.pivot_table(index=[*by, "seed"], columns=key, values=metric, aggfunc="first")


def paired_differences(
    runs: pd.DataFrame,
    metric: str,
    by: list[str],
    key: str = "tau",
    ref: float = 0.0,
    values: Sequence[float] | None = None,
    relative: bool = False,
) -> pd.DataFrame:
    """Per-seed ``metric(key=v) - metric(key=ref)`` for each v != ref, with 95% paired t-CIs.

    With ``relative=True`` the difference is divided by ``metric(key=ref)`` of the same seed.
    Returns one row per (by..., v): mean, ci_low, ci_high, n.
    """
    wide = paired_values(runs, metric, by, key)
    vals = [v for v in (values if values is not None else wide.columns) if v != ref]
    rows = []
    for v in vals:
        d = wide[v] - wide[ref]
        if relative:
            d = d / wide[ref]
        level = by[0] if len(by) == 1 else by  # scalar level for one key avoids a pandas warning
        grouped = d.groupby(level=level) if by else [((), d)]
        for g, series in grouped:
            g = g if isinstance(g, tuple) else (g,)
            rows.append((*g, v, *t_interval(series.to_numpy())))
    cols = [*by, key, "mean", "ci_low", "ci_high", "n"]
    out = pd.DataFrame(rows, columns=cols)
    out.insert(len(by) + 1, "metric", metric)
    return out


def ph_cut_effect(runs: pd.DataFrame, metrics: Sequence[str], by: list[str]) -> pd.DataFrame:
    """Per-seed ``metric(0.1%) - metric(0.6%)`` with 95% paired t-CIs."""
    parts = [
        paired_differences(runs, m, by, "tau", ref=PH_OLD, values=[PH_NEW])
        for m in metrics
        if m in runs
    ]
    return pd.concat(parts, ignore_index=True).drop(columns="tau")


def extreme_moves(
    runs: pd.DataFrame, returns: dict[int, np.ndarray], by: list[str], k: float = 5.0
) -> pd.Series:
    """Count of |R| > k * vol_ref per run; vol_ref = vol of the tau = 0 run, same seed and cell."""
    ref = runs[runs["tau"] == 0.0].set_index([*by, "seed"])["vol"]
    out = []
    for _, r in runs.iterrows():
        key = (*(r[b] for b in by), r["seed"])
        vref = ref.get(key if len(key) > 1 else key[0], math.nan)
        rr = returns.get(int(r["run_id"]))
        if rr is None or not np.isfinite(vref):
            out.append(math.nan)
        else:
            out.append(float(np.sum(np.abs(rr) > k * vref)))
    return pd.Series(out, index=runs.index, name="extreme_moves")


# --------------------------------------------------------------------------- hypotheses
@dataclass
class Verdict:
    hypothesis: str
    verdict: str  # "supported" | "not supported" | "mixed"
    n_comparisons: int
    text: str


def _ci(row: pd.Series, digits: int = 4) -> str:
    return f"{row['mean']:+.{digits}g} [{row['ci_low']:+.{digits}g}, {row['ci_high']:+.{digits}g}]"


def check_h1(runs: pd.DataFrame, by: list[str]) -> Verdict:
    """H1: the tax cuts chartist and noise fills per agent more than fundamentalist fills.

    For each cell (by..., tau > 0) the relative change in fills per agent vs tau = 0 is computed
    per type. The cell supports H1 if the CIs of both chartists and noise traders lie entirely
    below the fundamentalists' CI (non-overlapping, larger drop).
    """
    rel = {t: paired_differences(runs, f"fills_per_agent_{t}", by, relative=True)
           for t in TYPE_NAMES}  # fmt: skip
    f = rel["fundamentalist"].set_index([*by, "tau"])
    c = rel["chartist"].set_index([*by, "tau"])
    nz = rel["noise"].set_index([*by, "tau"])
    lines, hits = [], 0
    for idx in f.index:
        ok_c = c.loc[idx, "ci_high"] < f.loc[idx, "ci_low"]
        ok_n = nz.loc[idx, "ci_high"] < f.loc[idx, "ci_low"]
        hits += bool(ok_c and ok_n)
        lines.append(f"| {_idx(idx)} | {_ci(f.loc[idx])} | {_ci(c.loc[idx])} | "
                     f"{_ci(nz.loc[idx])} | {'yes' if ok_c and ok_n else 'no'} |")  # fmt: skip
    n = len(f.index)
    verdict = _verdict(hits, n)
    head = (f"| {', '.join([*by, 'tau'])} | fundamentalist | chartist | noise | H1 holds |\n"
            "|---|---|---|---|---|")  # fmt: skip
    text = (f"Relative change in fills per agent vs τ = 0 (mean [95% CI], paired by seed). "
            f"H1 holds in {hits} of {n} cells.\n\n{head}\n" + "\n".join(lines))  # fmt: skip
    return Verdict("H1", verdict, 2 * n, text)


def check_h2(runs: pd.DataFrame, shares: Sequence[float] = (0.40, 0.55)) -> Verdict:
    """H2: at chartist share 0.40 and 0.55, vol and mispricing fall (CI < 0) for some tau > 0.

    ASSUMPTION: both metrics must have CIs below 0 at the same tau; "supported" needs this at
    both shares, "mixed" at one share or for only one metric.
    """
    by = ["chartist_share"]
    dv = paired_differences(runs, "vol", by).set_index([*by, "tau"])
    dm = paired_differences(runs, "mispricing", by).set_index([*by, "tau"])
    lines, full, partial = [], 0, 0
    for s in shares:
        sub = [i for i in dv.index if np.isclose(i[0], s)]
        both = any(dv.loc[i, "ci_high"] < 0 and dm.loc[i, "ci_high"] < 0 for i in sub)
        one = any(dv.loc[i, "ci_high"] < 0 or dm.loc[i, "ci_high"] < 0 for i in sub)
        full += both
        partial += one and not both
        for i in sub:
            lines.append(f"| {i[0]:.2f} | {i[1]:g} | {_ci(dv.loc[i])} | {_ci(dm.loc[i])} |")
    verdict = "supported" if full == len(shares) else ("mixed" if full or partial else
                                                       "not supported")  # fmt: skip
    text = (
        "Paired differences vs τ = 0 (mean [95% CI]).\n\n"
        "| chartist share | τ | Δ vol | Δ mispricing |\n|---|---|---|---|\n" + "\n".join(lines)
    )
    n = 2 * sum(1 for i in dv.index if any(np.isclose(i[0], s) for s in shares))
    return Verdict("H2", verdict, n, text)


def check_h3(runs: pd.DataFrame, n_thin: int = 250) -> Verdict:
    """H3: at N = 250, vol rises (CI > 0) for some tau > 0, with spread up and depth down.

    "supported": some tau has all three; "mixed": vol rises but not both liquidity changes.
    """
    by = ["N"]
    d = {m: paired_differences(runs, m, by).set_index([*by, "tau"])
         for m in ("vol", "rel_spread_mean", "depth_mean")}  # fmt: skip
    rows = [i for i in d["vol"].index if i[0] == n_thin]
    lines, full, vol_up = [], False, False
    for i in rows:
        v = d["vol"].loc[i, "ci_low"] > 0
        sp = d["rel_spread_mean"].loc[i, "ci_low"] > 0
        dp = d["depth_mean"].loc[i, "ci_high"] < 0
        vol_up |= v
        full |= v and sp and dp
        lines.append(f"| {i[1]:g} | {_ci(d['vol'].loc[i])} | {_ci(d['rel_spread_mean'].loc[i])} | "
                     f"{_ci(d['depth_mean'].loc[i])} |")  # fmt: skip
    verdict = "supported" if full else ("mixed" if vol_up else "not supported")
    text = (
        f"N = {n_thin}: paired differences vs τ = 0 (mean [95% CI]).\n\n"
        "| τ | Δ vol | Δ rel. spread | Δ depth |\n|---|---|---|---|\n" + "\n".join(lines)
    )
    return Verdict("H3", verdict, 3 * len(rows), text)


def _idx(idx: object) -> str:
    parts = idx if isinstance(idx, tuple) else (idx,)
    return ", ".join(f"{p:g}" if isinstance(p, float) else str(p) for p in parts)


def _verdict(hits: int, n: int) -> str:
    if n and hits == n:
        return "supported"
    return "mixed" if hits else "not supported"


def hypotheses_markdown(verdicts: list[Verdict], experiment: str) -> str:
    """Render verdicts for ``results/<experiment>/analysis/hypotheses.md``."""
    out = [f"# Hypothesis checks: {experiment}", "",
           "Results describe the model's mechanism, not a forecast for the PSE. Each verdict "
           "lists how many confidence intervals it rests on; no correction for multiple "
           "comparisons is applied.", ""]  # fmt: skip
    for v in verdicts:
        out += [
            f"## {v.hypothesis}: {v.verdict}",
            "",
            f"Number of comparisons (CIs inspected): {v.n_comparisons}.",
            "",
            v.text,
            "",
        ]
    return "\n".join(out)


# --------------------------------------------------------------------------- E4
def sobol_outputs(runs: pd.DataFrame, outputs: Sequence[str]) -> pd.DataFrame:
    """PH cut effect ``Y = metric(0.1%) - metric(0.6%)`` per Sobol point (rows ordered by point)."""
    out = {}
    for m in outputs:
        wide = runs.pivot_table(index="point", columns="tau", values=m, aggfunc="first")
        out[m] = wide[PH_NEW] - wide[PH_OLD]
    return pd.DataFrame(out).sort_index()


def sobol_indices(
    y: pd.DataFrame, problem: dict[str, object], calc_second_order: bool = True
) -> pd.DataFrame:
    """First-order (S1) and total (ST) Sobol indices with 95% CIs for each output column.

    ASSUMPTION: a NaN output (a failed metric) is replaced by the mean of that output, since the
    Sobol estimator needs every sample point; the number replaced is reported in ``n_nan``.
    """
    from SALib.analyze import sobol

    rows = []
    for col in y.columns:
        v = y[col].to_numpy(dtype=float)
        n_nan = int(np.sum(~np.isfinite(v)))
        v = np.where(np.isfinite(v), v, np.nanmean(v))
        res = sobol.analyze(problem, v, calc_second_order=calc_second_order,
                            print_to_console=False, seed=0)  # fmt: skip
        for j, name in enumerate(problem["names"]):  # type: ignore[arg-type]
            rows.append({"output": col, "param": name, "S1": res["S1"][j],
                         "S1_conf": res["S1_conf"][j], "ST": res["ST"][j],
                         "ST_conf": res["ST_conf"][j], "n_nan": n_nan})  # fmt: skip
    return pd.DataFrame(rows)
