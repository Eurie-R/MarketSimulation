"""Plotting functions (matplotlib only) and the notebook sandbox widgets.

Figures illustrate the model's mechanism; they are not forecasts for the PSE.
"""

from __future__ import annotations

import io
import logging
import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from marketabm import metrics
from marketabm.agents import TYPE_NAMES
from marketabm.config import ModelConfig, mix_from_chartist_share, with_overrides
from marketabm.market import LiveFrame, Simulation
from marketabm.results import SimulationResult
from marketabm.stylized_facts import moments, standardize, validation_checks

log = logging.getLogger(__name__)

# Categorical slots 1-3 of the reference palette (validated all-pairs for 3 series).
TYPE_COLORS = {"fundamentalist": "#2a78d6", "chartist": "#eb6834", "noise": "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a8a69e"
GRID = "#e6e5e0"
PASS_C, FAIL_C = "#008300", "#e34948"
PH_OLD, PH_NEW = 0.006, 0.001


def _style(ax: Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.title.set_color(INK)
    ax.title.set_fontsize(10)


def pct(x: float) -> str:
    """Format a tax fraction as a percent label (0.001 -> '0.1%')."""
    return f"{x * 100:.2f}".rstrip("0").rstrip(".") + "%"


def _shade_burn(ax: Axes, T_burn: int) -> None:  # noqa: N803
    if T_burn > 0:
        ax.axvspan(0, T_burn, color=GRID, alpha=0.6, lw=0, label="burn-in")


# ---------------------------------------------------------------- single-run panels
def plot_price_vs_fundamental(r: SimulationResult, ax: Axes, label: str = "price") -> None:
    """F1: market price vs fundamental value, burn-in shaded."""
    t = np.arange(len(r.price))
    _shade_burn(ax, r.config.T_burn)
    ax.plot(t, r.fundamental, color=INK_2, lw=1.2, ls="--", label="fundamental value", zorder=3)
    ax.plot(t, r.price, color=TYPE_COLORS["fundamentalist"], lw=1.0, label=label)
    ax.set(xlabel="step", ylabel="price (currency)", title="Price vs fundamental value")
    if np.nanmax(r.price) / np.nanmin(r.price) > 20:
        ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _style(ax)


def plot_period_returns(r: SimulationResult, ax: Axes) -> None:
    R = r.period_returns()  # noqa: N806
    ax.plot(np.arange(len(R)), R, color=INK_2, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(
        xlabel=f"period ({r.config.return_interval} steps), after burn-in",
        ylabel="log return",
        title="Period returns",
    )
    _style(ax)


def plot_spread(r: SimulationResult, ax: Axes) -> None:
    mid = (r.best_bid + r.best_ask) / 2.0
    rel = (r.best_ask - r.best_bid) / mid * 100.0
    ax.plot(np.arange(len(rel)), rel, color=TYPE_COLORS["fundamentalist"], lw=0.6)
    _shade_burn(ax, r.config.T_burn)
    ax.set(xlabel="step", ylabel="relative spread (%)", title="Bid-ask spread")
    _style(ax)


def plot_volume(r: SimulationResult, ax: Axes) -> None:
    w = r.config.return_interval
    n = len(r.volume) // w
    v = r.volume[: n * w].reshape(n, w).sum(axis=1)
    ax.bar(np.arange(n) * w, v, width=w, color=MUTED, linewidth=0)
    _shade_burn(ax, r.config.T_burn)
    ax.set(xlabel="step", ylabel=f"units per {w} steps", title="Traded volume")
    _style(ax)


def plot_volume_share(r: SimulationResult, ax: Axes) -> None:
    s = r.summary()
    shares = [s.get(f"volume_share_{k}", math.nan) for k in TYPE_NAMES]
    shares = [0.0 if math.isnan(x) else x for x in shares]
    y = np.arange(3)
    ax.barh(y, shares, color=[TYPE_COLORS[k] for k in TYPE_NAMES], height=0.6)
    for yi, val in zip(y, shares, strict=True):
        ax.text(val + 0.01, yi, f"{val:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, TYPE_NAMES)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set(xlabel="share of fill participations", title="Volume share by trader type")
    _style(ax)


def plot_acf(R: np.ndarray, ax: Axes, absolute: bool, label: str = "model") -> None:  # noqa: N803
    x = np.abs(R) if absolute else R
    a = metrics.acf(x, 50)
    band = metrics.acf_band(len(R))
    lags = np.arange(1, 51)
    ax.axhspan(-band, band, color=GRID, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.vlines(lags, 0, a[1:], color=TYPE_COLORS["fundamentalist"], lw=1.5, label=label)
    name = "|R|" if absolute else "R"
    ax.set(xlabel="lag (periods)", ylabel="autocorrelation", title=f"ACF of {name}")
    _style(ax)


def plot_return_hist(R: np.ndarray, ax: Axes) -> None:  # noqa: N803
    z = standardize(R)
    ax.hist(z, bins=60, density=True, color=MUTED, linewidth=0)
    xs = np.linspace(z.min(), z.max(), 200)
    ax.plot(xs, np.exp(-(xs**2) / 2) / math.sqrt(2 * math.pi), color=INK, lw=1.2,
            label="standard normal")  # fmt: skip
    ax.set_yscale("log")
    ax.set(xlabel="standardized period return", ylabel="density (log)", title="Return distribution")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)


def stylized_facts_card(R: np.ndarray, ax: Axes) -> None:  # noqa: N803
    """Text card: moments and pass/fail against the validation targets."""
    ax.axis("off")
    if len(R) < 60:
        ax.text(0, 0.5, "Too few period returns for stylized facts", color=INK)
        return
    m = moments(R)
    checks = validation_checks(R)
    lines = [
        ("excess kurtosis", f"{m['excess_kurtosis']:.2f}"),
        ("Hill alpha (top 5%)", f"{m['hill_alpha']:.2f}"),
        ("ACF R lag 1", f"{m['acf_r_lag1']:+.3f}"),
        ("mean ACF |R| lags 1-10", f"{m['acf_absr_1_10']:+.3f}"),
        ("mean ACF |R| lags 11-50", f"{m['acf_absr_11_50']:+.3f}"),
    ]
    ax.set_title("Stylized facts", loc="left", fontsize=10, color=INK)
    y = 0.92
    for name, val in lines:
        ax.text(0.0, y, name, fontsize=9, color=INK_2, transform=ax.transAxes)
        ax.text(0.98, y, val, fontsize=9, color=INK, ha="right", transform=ax.transAxes)
        y -= 0.1
    y -= 0.04
    for name, ok in checks.items():
        mark = "PASS" if ok else "FAIL"
        ax.text(0.0, y, mark, fontsize=8, weight="bold", color=PASS_C if ok else FAIL_C,
                transform=ax.transAxes)  # fmt: skip
        ax.text(0.16, y, name, fontsize=8, color=INK_2, transform=ax.transAxes)
        y -= 0.09


def plot_run(r: SimulationResult) -> Figure:
    """Full single-run dashboard used by the sandbox."""
    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)
    plot_price_vs_fundamental(r, fig.add_subplot(gs[0, :]))
    plot_period_returns(r, fig.add_subplot(gs[1, 0:2]))
    plot_volume_share(r, fig.add_subplot(gs[1, 2]))
    plot_spread(r, fig.add_subplot(gs[2, 0]))
    plot_volume(r, fig.add_subplot(gs[2, 1]))
    plot_return_hist(r.period_returns(), fig.add_subplot(gs[2, 2]))
    R = r.period_returns()  # noqa: N806
    plot_acf(R, fig.add_subplot(gs[3, 0]), absolute=False)
    plot_acf(R, fig.add_subplot(gs[3, 1]), absolute=True)
    stylized_facts_card(R, fig.add_subplot(gs[3, 2]))
    c = r.config
    fig.suptitle(
        f"seed {r.seed} · tax {pct(c.tau)} · N={c.N} · mix F/C/N = "
        f"{c.mix.fundamentalist:.2f}/{c.mix.chartist:.2f}/{c.mix.noise:.2f}"
        f"{' · switching on' if c.switching.enabled else ''}",
        color=INK,
        fontsize=11,
    )
    return fig


def plot_compare(ra: SimulationResult, rb: SimulationResult) -> Figure:
    """Same seed at two tax rates on shared axes (demonstrates common random numbers)."""
    ca, cb = TYPE_COLORS["fundamentalist"], TYPE_COLORS["chartist"]
    la, lb = f"tax {pct(ra.config.tau)}", f"tax {pct(rb.config.tau)}"
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    ax = axes[0, 0]
    t = np.arange(len(ra.price))
    _shade_burn(ax, ra.config.T_burn)
    ax.plot(t, ra.fundamental, color=INK_2, ls="--", lw=1.2, label="fundamental (shared)", zorder=3)
    ax.plot(t, ra.price, color=ca, lw=0.9, label=la)
    ax.plot(t, rb.price, color=cb, lw=0.9, label=lb)
    ax.set(xlabel="step", ylabel="price (currency)", title="Price paths, same seed")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)

    sa, sb = ra.summary(), rb.summary()
    keys = [
        ("vol", "volatility (sd of R)"),
        ("mispricing", "mean |log P - log f|"),
        ("rel_spread_mean", "mean relative spread"),
        ("volume_total", "volume after burn-in"),
    ]
    ax = axes[0, 1]
    ax.axis("off")
    ax.set_title("Summary after burn-in", loc="left", fontsize=10, color=INK)
    ax.text(0.45, 0.9, la, color=INK, fontsize=9, weight="bold", transform=ax.transAxes)
    ax.text(0.75, 0.9, lb, color=INK, fontsize=9, weight="bold", transform=ax.transAxes)
    for j, (k, name) in enumerate(keys + [("tax_revenue", "tax revenue")]):
        y = 0.78 - j * 0.12
        ax.text(0.0, y, name, fontsize=9, color=INK_2, transform=ax.transAxes)
        ax.text(0.45, y, f"{sa[k]:.4g}", fontsize=9, color=INK, transform=ax.transAxes)
        ax.text(0.75, y, f"{sb[k]:.4g}", fontsize=9, color=INK, transform=ax.transAxes)

    ax = axes[1, 0]
    ys = np.arange(3)
    for off, s, col, lab in ((-0.18, sa, ca, la), (0.18, sb, cb, lb)):
        vals = [s.get(f"skip_rate_tax_{k}", math.nan) for k in TYPE_NAMES]
        ax.barh(ys + off, np.nan_to_num(vals), height=0.34, color=col, label=lab)
    ax.set_yticks(ys, TYPE_NAMES)
    ax.invert_yaxis()
    ax.set(xlabel="share of activations skipped due to tax", title="Tax-induced skips by type")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)

    ax = axes[1, 1]
    for r, col, lab in ((ra, ca, la), (rb, cb, lb)):
        R = r.period_returns()  # noqa: N806
        a = metrics.acf(np.abs(R), 50)
        ax.plot(np.arange(1, 51), a[1:], color=col, lw=1.5, marker="o", ms=3, label=lab)
    ax.axhspan(-metrics.acf_band(len(R)), metrics.acf_band(len(R)), color=GRID, lw=0)
    ax.set(
        xlabel="lag (periods)", ylabel="autocorrelation", title="ACF of |R| (volatility clustering)"
    )
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    return fig


# ---------------------------------------------------------------- sandbox
def sandbox_config(
    base: ModelConfig,
    *,
    tau: float,
    shares: tuple[float, float, float],
    N: int,  # noqa: N803
    T: int,  # noqa: N803
    switching: bool = False,
    beta: float | None = None,
    price_band: float | None = None,
    order_size: str | None = None,
) -> ModelConfig:
    """Build a sandbox config: shares are normalized to sum to 1.

    ASSUMPTION: the sandbox sets T_burn = 10% of T and rounds T and T_burn to multiples of
    return_interval, so that short runs (default T = 10,000) keep most of their steps.
    """
    tot = sum(shares)
    if tot <= 0:
        raise ValueError("at least one type share must be positive")
    f, c, n = (s / tot for s in shares)
    ri = base.return_interval
    T = max(10 * ri, (T // ri) * ri)  # noqa: N806
    T_burn = (T // 10 // ri) * ri  # noqa: N806
    sw = {"enabled": bool(switching), "beta": beta if switching else base.switching.beta}
    return with_overrides(
        base,
        {
            "tau": tau,
            "N": N,
            "M": min(base.M, N),
            "T": T,
            "T_burn": T_burn,
            # fundamentalists absorb floating error so the shares sum to exactly 1
            "mix": {"fundamentalist": 1.0 - c - n, "chartist": c, "noise": n},
            "switching": sw,
            "price_band": base.price_band if price_band is None else price_band,
            "order_size": order_size or base.order_size,
        },
    )


def safe_run(cfg: ModelConfig, seed: int) -> SimulationResult | None:
    """Run and report divergence instead of raising (for interactive use)."""
    from marketabm.market import run

    try:
        return run(cfg, seed)
    except OverflowError:
        print(
            "The market diverged (price overflow). This can happen with price_band = None "
            "(no order-price band); set a band or reduce the chartist share."
        )
        return None


def sandbox_ui(base: ModelConfig) -> None:
    """Interactive single-run explorer (ipywidgets, interact_manual)."""
    import ipywidgets as w
    from IPython.display import display

    style = {"description_width": "150px"}
    lay = w.Layout(width="460px")
    ctrl = {
        "tau_pct": w.FloatSlider(0.0, min=0.0, max=1.0, step=0.05, description="tax τ (%)",
                                 readout_format=".2f", style=style, layout=lay),
        "fund": w.FloatSlider(base.mix.fundamentalist, min=0, max=1, step=0.05,
                              description="fundamentalist share", style=style, layout=lay),
        "chart": w.FloatSlider(base.mix.chartist, min=0, max=1, step=0.05,
                               description="chartist share", style=style, layout=lay),
        "noise": w.FloatSlider(base.mix.noise, min=0, max=1, step=0.05,
                               description="noise share", style=style, layout=lay),
        "N": w.IntSlider(base.N, min=50, max=2000, step=50, description="agents N",
                         style=style, layout=lay),
        "T": w.IntSlider(10_000, min=2_000, max=50_000, step=1_000, description="steps T",
                         style=style, layout=lay),
        "band_pct": w.FloatSlider((base.price_band or 0.01) * 100, min=0.25, max=5, step=0.25,
                                  description="price band (%)", style=style, layout=lay),
        "order_size": w.Dropdown(options=["signal", "unit"], value=base.order_size,
                                 description="order size", style=style),
        "seed": w.IntText(1, description="seed", style=style),
        "switching": w.Checkbox(False, description="strategy switching"),
        "beta": w.FloatText(2.0, description="switching β (E3: 2 mild, 15 strong)", style=style),
    }  # fmt: skip

    def _go(tau_pct, fund, chart, noise, N, T, band_pct, order_size, seed, switching, beta):  # noqa: N803
        cfg = sandbox_config(base, tau=tau_pct / 100, shares=(fund, chart, noise), N=N, T=T,
                             switching=switching, beta=beta, price_band=band_pct / 100,
                             order_size=order_size)  # fmt: skip
        r = safe_run(cfg, seed)
        if r is None:
            return
        plot_run(r)
        plt.show()
        print(f"run time: {r.metadata['runtime_s']:.1f} s")

    display(w.interactive(_go, {"manual": True, "manual_name": "Run"}, **ctrl))


def compare_ui(base: ModelConfig) -> None:
    """Run one seed at two tax rates and plot them together."""
    import ipywidgets as w
    from IPython.display import display

    style = {"description_width": "150px"}
    lay = w.Layout(width="460px")
    ctrl = {
        "tau_a_pct": w.FloatSlider(PH_OLD * 100, min=0, max=1, step=0.05,
                                   description="tax A (%)", style=style, layout=lay),
        "tau_b_pct": w.FloatSlider(PH_NEW * 100, min=0, max=1, step=0.05,
                                   description="tax B (%)", style=style, layout=lay),
        "chart": w.FloatSlider(base.mix.chartist, min=0, max=1, step=0.05,
                               description="chartist share", style=style, layout=lay),
        "N": w.IntSlider(base.N, min=50, max=2000, step=50, description="agents N",
                         style=style, layout=lay),
        "T": w.IntSlider(10_000, min=2_000, max=50_000, step=1_000, description="steps T",
                         style=style, layout=lay),
        "seed": w.IntText(1, description="seed", style=style),
    }  # fmt: skip

    def _go(tau_a_pct, tau_b_pct, chart, N, T, seed):  # noqa: N803
        mix = mix_from_chartist_share(chart, base.mix)
        f, n = mix["fundamentalist"], mix["noise"]
        runs = []
        for tau in (tau_a_pct / 100, tau_b_pct / 100):
            cfg = sandbox_config(base, tau=tau, shares=(f, chart, n), N=N, T=T)
            r = safe_run(cfg, seed)
            if r is None:
                return
            runs.append(r)
        plot_compare(*runs)
        plt.show()

    display(w.interactive(_go, {"manual": True, "manual_name": "Run"}, **ctrl))


# ---------------------------------------------------------------- live view
BID_C, ASK_C = "#2f8f4e", "#d9534f"
LIVE_TAX_STEPS = (0.0, 0.1, 0.25, 0.6, 1.0)  # percent; old/new PH rates included


def live_figure() -> tuple[Figure, dict[str, Axes]]:
    """Figure and axes for the live market view (not attached to pyplot)."""
    fig = Figure(figsize=(12, 7), dpi=80, layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], width_ratios=[1.25, 1.25, 1])
    axes = {
        "price": fig.add_subplot(gs[0, :2]),
        "book": fig.add_subplot(gs[0, 2]),
        "mis": fig.add_subplot(gs[1, 0]),
        "vol": fig.add_subplot(gs[1, 1]),
        "who": fig.add_subplot(gs[1, 2]),
    }
    return fig, axes


def draw_live(axes: dict[str, Axes], fr: LiveFrame, cfg: ModelConfig) -> None:
    """Redraw every live panel from one :class:`~marketabm.market.LiveFrame`."""
    for ax in axes.values():
        ax.clear()
    x = np.arange(fr.t_start, fr.t + 1)
    _live_price(axes["price"], x, fr, cfg)
    _live_book(axes["book"], fr)
    _live_mispricing(axes["mis"], x, fr)
    _live_volume(axes["vol"], x, fr)
    _live_who(axes["who"], fr)
    for ax in axes.values():
        _style(ax)


def _live_price(ax: Axes, x: np.ndarray, fr: LiveFrame, cfg: ModelConfig) -> None:
    if fr.t_start < cfg.T_burn:
        ax.axvspan(fr.t_start, min(cfg.T_burn, fr.t), color=GRID, alpha=0.6, lw=0,
                   label="burn-in")  # fmt: skip
    # Gaps (one-sided book) are bridged for drawing only, so the band is one polygon.
    bid, ask = _ffill(fr.best_bid), _ffill(fr.best_ask)
    ax.fill_between(x, bid, ask, color=MUTED, alpha=0.35, lw=0, label="bid–ask spread")
    ax.plot(x, fr.price, color=INK, lw=0.7, label="last trade price")
    ax.plot(x, fr.fundamental, color=TYPE_COLORS["fundamentalist"], lw=1.8,
            label="fundamental value", zorder=3)  # fmt: skip
    ax.plot([fr.t], [fr.price[-1]], "o", color=INK, ms=4)
    ax.set_xlim(x[0], max(x[-1], x[0] + 1))
    ax.set_title(f"Price vs fundamental value (τ = {pct(cfg.tau)})", loc="left")
    ax.set_xlabel("step")
    ax.set_ylabel("price (currency units)")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)


def _live_book(ax: Axes, fr: LiveFrame) -> None:
    prices = [p for p, _ in fr.bids + fr.asks]
    width = _bin_width(prices)
    for levels, color, label in ((fr.bids, BID_C, "bids (buy)"), (fr.asks, ASK_C, "asks (sell)")):
        if levels:
            centers, qty = _bin_levels(levels, width)
            ax.barh(centers, qty, height=0.9 * width, color=color, label=label)
    ax.axhline(fr.price[-1], color=INK, lw=0.8, ls=":", label="last price")
    ax.set_title(f"Order book: {fr.book_size} resting orders", loc="left")
    ax.set_xlabel(f"resting quantity (units per {width:.2f} price bin)")
    ax.set_ylabel("limit price")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs (leading NaNs take the first finite value); display only."""
    ok = np.isfinite(a)
    if not ok.any():
        return a
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[: int(np.argmax(ok))] = a[int(np.argmax(ok))]
    return out


def _bin_width(prices: list[float], n_bins: int = 30) -> float:
    """Price bin for the book chart: about ``n_bins`` bins over the shown levels."""
    if len(prices) < 2:
        return 0.05
    span = max(prices) - min(prices)
    return max(0.01, round(span / n_bins, 2)) if span > 0 else 0.05


def _bin_levels(levels: list[tuple[float, int]], width: float) -> tuple[np.ndarray, np.ndarray]:
    """Sum quantities of ``(price, qty)`` levels into bins of ``width``; returns centers, qty."""
    p = np.array([lv[0] for lv in levels])
    q = np.array([lv[1] for lv in levels], dtype=float)
    keys, inv = np.unique(np.floor(p / width).astype(np.int64), return_inverse=True)
    return (keys + 0.5) * width, np.bincount(inv, weights=q)


def _live_mispricing(ax: Axes, x: np.ndarray, fr: LiveFrame) -> None:
    gap = 100 * np.log(fr.price / fr.fundamental)
    ax.axhline(0, color=MUTED, lw=0.8)
    # Clipped fills (one polygon each) instead of where= masks, which render slowly.
    ax.fill_between(x, np.maximum(gap, 0), color=ASK_C, alpha=0.35, lw=0, label="overpriced")
    ax.fill_between(x, np.minimum(gap, 0), color=BID_C, alpha=0.35, lw=0, label="underpriced")
    ax.plot(x, gap, color=INK_2, lw=0.6)
    ax.set_xlim(x[0], max(x[-1], x[0] + 1))
    ax.set_title("Mispricing log(P / f)", loc="left")
    ax.set_xlabel("step")
    ax.set_ylabel("% from fundamental")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def _live_volume(ax: Axes, x: np.ndarray, fr: LiveFrame) -> None:
    w = min(20, len(fr.volume))
    smooth = np.convolve(fr.volume, np.ones(w) / w, mode="same") if w else fr.volume
    # One filled area instead of one bar per step: thousands of bar patches render slowly.
    ax.fill_between(x, fr.volume, step="mid", color=GRID, lw=0, label="per step")
    ax.plot(x, smooth, color=INK_2, lw=1.0, label=f"{w}-step average")
    ax.set_xlim(x[0], max(x[-1], x[0] + 1))
    ax.set_title("Traded volume", loc="left")
    ax.set_xlabel("step")
    ax.set_ylabel("units per step")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def _live_who(ax: Axes, fr: LiveFrame) -> None:
    fills = fr.fills_by_type.sum(axis=0).astype(float)
    active = fr.active_by_type.sum(axis=0).astype(float)
    skip = fr.skipped_tax_by_type.sum(axis=0).astype(float)
    share = 100 * fills / fills.sum() if fills.sum() else np.zeros(3)
    skip_rate = 100 * np.divide(skip, active, out=np.zeros(3), where=active > 0)
    y = np.arange(3)
    colors = [TYPE_COLORS[n] for n in TYPE_NAMES]
    ax.barh(y - 0.2, share, height=0.38, color=colors)
    ax.barh(y + 0.2, skip_rate, height=0.38, color=colors, alpha=0.35, hatch="///")
    for yi, a, b in zip(y, share, skip_rate, strict=True):
        ax.text(a + 1, yi - 0.2, f"{a:.0f}%", va="center", fontsize=8, color=INK_2)
        ax.text(b + 1, yi + 0.2, f"{b:.0f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, list(TYPE_NAMES))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% (within the shown window)")
    ax.set_title("Who trades, who the tax stops", loc="left")
    handles = [Patch(color=MUTED, label="share of all trades"),
               Patch(facecolor="white", edgecolor=MUTED, hatch="///",
                     label="activations skipped because of tax")]  # fmt: skip
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)


class LiveViewer:
    """Live market view for the notebook: the real simulator stepped in a background thread.

    Each frame advances the simulation by ``speed`` steps, redraws the panels into a PNG and
    pushes it to an ``ipywidgets.Image``. Reset applies the parameter widgets.
    """

    def __init__(self, base: ModelConfig) -> None:
        import ipywidgets as w

        self.base = base
        self.fig, self.axes = live_figure()
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self.sim: Simulation | None = None
        style = {"description_width": "120px"}
        lay = w.Layout(width="330px")
        self.w = {
            "tau": w.SelectionSlider(options=[(f"{v:g}%", v / 100) for v in LIVE_TAX_STEPS],
                                     value=0.0, description="tax τ", style=style, layout=lay),
            "chart": w.FloatSlider(base.mix.chartist, min=0, max=0.8, step=0.05,
                                   description="chartist share", style=style, layout=lay),
            "N": w.IntSlider(base.N, min=100, max=2000, step=50, description="agents N",
                             style=style, layout=lay),
            "T": w.IntSlider(20_000, min=2_000, max=50_000, step=1_000, description="steps T",
                             style=style, layout=lay),
            "seed": w.IntText(1, description="seed", style=style, layout=lay),
            "speed": w.IntSlider(50, min=5, max=1000, step=5, description="steps / frame",
                                 style=style, layout=lay),
            "window": w.IntSlider(2000, min=200, max=20_000, step=200,
                                  description="window (steps)", style=style, layout=lay),
        }  # fmt: skip
        self.play = w.Button(description="▶ Play", button_style="success")
        self.reset_btn = w.Button(description="⟲ Reset with settings")
        self.status = w.HTML()
        self.image = w.Image(format="png")
        self.play.on_click(lambda _: self.toggle())
        self.reset_btn.on_click(lambda _: self.reset())

    # -- widget layout
    def widget(self) -> Any:
        import ipywidgets as w

        model = w.VBox(
            [
                w.HTML("<b>Model (applied on reset)</b>"),
                *(self.w[k] for k in ("tau", "chart", "N", "T", "seed")),
            ]
        )
        view = w.VBox([w.HTML("<b>View (live)</b>"), self.w["speed"], self.w["window"],
                       w.HBox([self.play, self.reset_btn])])  # fmt: skip
        return w.VBox([w.HBox([model, view]), self.status, self.image])

    def config(self) -> ModelConfig:
        mix = mix_from_chartist_share(self.w["chart"].value, self.base.mix)
        shares = (mix["fundamentalist"], mix["chartist"], mix["noise"])
        return sandbox_config(self.base, tau=self.w["tau"].value, shares=shares,
                              N=self.w["N"].value, T=self.w["T"].value)  # fmt: skip

    # -- control
    def reset(self) -> None:
        self.pause()
        self.status.value = "Preparing run (drawing random streams)…"
        with self._lock:
            self.sim = Simulation(self.config(), int(self.w["seed"].value))
            self.sim.advance(1)
        self.render()

    def toggle(self) -> None:
        if self._running.is_set():
            self.pause()
            return
        if self.sim is None or self.sim.done:
            self.reset()
        self._running.set()
        self.play.description, self.play.button_style = "⏸ Pause", "warning"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._running.clear()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self._thread = None
        self.play.description, self.play.button_style = "▶ Play", "success"

    def _loop(self) -> None:
        try:
            while self._running.is_set() and self.sim is not None and not self.sim.done:
                with self._lock:
                    self.sim.advance(int(self.w["speed"].value))
                self.render()
        except OverflowError:
            self.status.value = "<b>The market diverged (price overflow).</b> Reset to try again."
        finally:
            self._running.clear()
            self.play.description, self.play.button_style = "▶ Play", "success"

    # -- drawing
    def render(self) -> None:
        sim = self.sim
        if sim is None:
            return
        with self._lock:
            fr = sim.frame(int(self.w["window"].value))
        draw_live(self.axes, fr, sim.cfg)
        buf = io.BytesIO()
        FigureCanvasAgg(self.fig).print_png(buf)
        self.image.value = buf.getvalue()
        self.status.value = live_status(fr, sim.cfg)


def live_status(fr: LiveFrame, cfg: ModelConfig) -> str:
    """One-line HTML summary of the current market state."""
    p, f = fr.price[-1], fr.fundamental[-1]
    bb, ba = fr.best_bid[-1], fr.best_ask[-1]
    spread = f"{100 * (ba - bb) / ((ba + bb) / 2):.2f}%" if np.isfinite(bb + ba) else "one-sided"
    phase = "burn-in" if fr.t < cfg.T_burn else "measured"
    done = " — <b>finished</b>" if fr.t >= cfg.T - 1 else ""
    return (
        f"<b>step {fr.t + 1:,} / {cfg.T:,}</b> ({phase}){done} · price {p:.2f} · "
        f"fundamental {f:.2f} · gap {100 * math.log(p / f):+.2f}% · spread {spread} · "
        f"tax revenue {fr.tax_revenue:,.2f}"
    )


def live_ui(base: ModelConfig) -> LiveViewer:
    """Display the live market view and return its controller."""
    from IPython.display import display

    viewer = LiveViewer(base)
    display(viewer.widget())
    viewer.reset()
    return viewer


# ---------------------------------------------------------------- experiment figures (14.1)
METRIC_LABELS = {
    "vol": "volatility (sd of period returns)",
    "mispricing": "mispricing (mean |log P − log f|)",
    "rel_spread_mean": "relative spread",
    "depth_mean": "depth within ±1% (units)",
    "volume_total": "volume traded (units)",
    "tax_revenue": "tax revenue (currency)",
}
# One colormap per metric family: price risk, liquidity, activity, revenue.
METRIC_CMAPS = {
    "vol": "OrRd", "mispricing": "OrRd", "rel_spread_mean": "PuBu", "depth_mean": "PuBu",
    "volume_total": "BuGn", "tax_revenue": "YlGn",
}  # fmt: skip


def save_figure(fig: Figure, directory: str | Path, name: str) -> list[Path]:
    """Save as PNG (200 dpi) and PDF; returns the paths."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / f"{name}.png", d / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    return paths


def _tau_pos(taus: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Evenly spaced x positions for tax rates (the grid is uneven: 0, 0.1, 0.25, 0.6, 1%)."""
    order = list(taus)
    return np.array([order.index(min(order, key=lambda t: abs(t - v))) for v in values])


def _tau_axis(ax: Axes, taus: Sequence[float]) -> None:
    """Categorical percent ticks for tau, with markers for the old and new PH rates."""
    ax.set_xticks(range(len(taus)), [pct(t) for t in taus])
    for t, name in ((PH_OLD, "old PH"), (PH_NEW, "new PH")):
        hit = [i for i, v in enumerate(taus) if np.isclose(v, t)]
        if hit:
            ax.axvline(hit[0], color=MUTED, lw=0.8, ls=":", zorder=0)
            ax.annotate(name, (hit[0], 1.0), xycoords=("data", "axes fraction"),
                        fontsize=7, color=INK_2, ha="center", va="bottom")  # fmt: skip
    ax.set_xlabel("transaction tax τ (% of sale value; evenly spaced)")


def fig_price_path(r: SimulationResult) -> Figure:
    """F1: price vs fundamental value over time for one run, burn-in shaded."""
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    plot_price_vs_fundamental(r, ax)
    ax.set_title(f"F1. Price vs fundamental value (seed {r.seed}, τ = {pct(r.config.tau)})")
    return fig


def fig_stylized_facts(r_model: np.ndarray, r_emp: np.ndarray, emp_label: str) -> Figure:
    """F2: return histogram vs normal (log y), ACF of R and of |R|; model vs empirical."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    series = (("model", standardize(r_model), TYPE_COLORS["fundamentalist"]),
              (emp_label, standardize(r_emp), TYPE_COLORS["chartist"]))  # fmt: skip
    bins = np.linspace(-8, 8, 81)
    for label, z, color in series:
        axes[0].hist(z, bins=bins, density=True, histtype="step", color=color, lw=1.4,
                     label=label)  # fmt: skip
    xs = np.linspace(-8, 8, 400)
    axes[0].plot(xs, np.exp(-(xs**2) / 2) / math.sqrt(2 * math.pi), color=INK, lw=1,
                 ls="--", label="standard normal")  # fmt: skip
    axes[0].set(yscale="log", ylim=(1e-4, 1), xlabel="standardized return",
                ylabel="density (log scale)", title="Return distribution")  # fmt: skip
    for ax, absolute in ((axes[1], False), (axes[2], True)):
        for k, (label, z, color) in enumerate(series):
            a = metrics.acf(np.abs(z) if absolute else z, 50)
            lags = np.arange(1, 51) + (k - 0.5) * 0.3
            ax.vlines(lags, 0, a[1:], color=color, lw=1.4, label=label)
            band = metrics.acf_band(len(z))
            ax.axhspan(-band, band, color=color, alpha=0.08, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        name = "|R|" if absolute else "R"
        ax.set(xlabel="lag (model periods / trading days)", ylabel="autocorrelation",
               title=f"ACF of {name} (shaded: 95% band)")  # fmt: skip
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
    fig.suptitle(f"F2. Stylized facts: model (calibrated) vs {emp_label}", fontsize=11)
    return fig


def fig_heatmaps(cm: pd.DataFrame, metric_names: Sequence[str]) -> Figure:
    """F3: heatmaps of cell means over tau x chartist share."""
    fig, axes = plt.subplots(1, len(metric_names), figsize=(3.6 * len(metric_names), 3.8),
                             layout="constrained")  # fmt: skip
    for ax, m in zip(np.atleast_1d(axes), metric_names, strict=True):
        sub = cm[cm["metric"] == m].pivot(index="chartist_share", columns="tau", values="mean")
        im = ax.imshow(sub.to_numpy(), cmap=METRIC_CMAPS.get(m, "viridis"), aspect="auto",
                       origin="lower")  # fmt: skip
        ax.set_xticks(range(sub.shape[1]), [pct(t) for t in sub.columns], fontsize=8)
        ax.set_yticks(range(sub.shape[0]), [f"{c:.0%}" for c in sub.index], fontsize=8)
        for j, t in enumerate(sub.columns):
            if np.isclose(t, PH_OLD) or np.isclose(t, PH_NEW):
                ax.get_xticklabels()[j].set_fontweight("bold")
        for (i, j), v in np.ndenumerate(sub.to_numpy()):
            ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=7, color=INK)
        ax.set(xlabel="tax τ (bold: old / new PH rate)", ylabel="chartist share")
        ax.set_title(METRIC_LABELS.get(m, m), fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"F3. Cell means over tax rate and trader mix ({int(cm['n'].max())} seeds)",
                 fontsize=11)  # fmt: skip
    return fig


def fig_ph_cut(ph: pd.DataFrame, metric_names: Sequence[str]) -> Figure:
    """F4: PH cut effect metric(0.1%) − metric(0.6%) with 95% CIs, x = chartist share."""
    n = len(metric_names)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.6), layout="constrained")
    for ax, m in zip(np.atleast_1d(axes), metric_names, strict=True):
        sub = ph[ph["metric"] == m].sort_values("chartist_share")
        x = sub["chartist_share"].to_numpy()
        y = sub["mean"].to_numpy()
        err = np.vstack([y - sub["ci_low"], sub["ci_high"] - y])
        colors = [ASK_C if lo > 0 else (BID_C if hi < 0 else MUTED)
                  for lo, hi in zip(sub["ci_low"], sub["ci_high"], strict=True)]  # fmt: skip
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=INK_2, capsize=3, lw=1)
        ax.scatter(x, y, c=colors, s=36, zorder=3)
        ax.set_xticks(x, [f"{c:.0%}" for c in x])
        ax.set(xlabel="chartist share", ylabel=f"Δ {m}")
        ax.set_title(METRIC_LABELS.get(m, m), fontsize=9)
        _style(ax)
    fig.suptitle("F4. Model effect of the PH tax cut: metric(τ = 0.1%) − metric(τ = 0.6%), "
                 "mean and 95% paired CI (red: CI > 0, green: CI < 0)", fontsize=10)  # fmt: skip
    return fig


def fig_skip_rates(cm: pd.DataFrame) -> Figure:
    """F5: share of activations skipped because of the tax, by type, vs tau (H1)."""
    shares = sorted(cm["chartist_share"].unique())
    taus = sorted(cm["tau"].unique())
    fig, axes = plt.subplots(1, len(shares), figsize=(3.4 * len(shares), 3.6), sharey=True,
                             layout="constrained")  # fmt: skip
    for ax, c in zip(np.atleast_1d(axes), shares, strict=True):
        for name in TYPE_NAMES:
            sel = (cm["metric"] == f"skip_rate_tax_{name}") & np.isclose(cm["chartist_share"], c)
            sub = cm[sel].sort_values("tau")
            x = _tau_pos(taus, sub["tau"])
            ax.plot(x, 100 * sub["mean"], "o-", color=TYPE_COLORS[name], ms=4, label=name)
            ax.fill_between(x, 100 * sub["ci_low"], 100 * sub["ci_high"],
                            color=TYPE_COLORS[name], alpha=0.2, lw=0)  # fmt: skip
        _tau_axis(ax, taus)
        ax.set_title(f"chartist share {c:.0%}", fontsize=9, pad=12)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("activations skipped because of tax (%)")
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.suptitle("F5. Who the tax stops: skip rate by trader type (mean, 95% CI)", fontsize=11)
    return fig


def fig_thin_market(cm: pd.DataFrame) -> Figure:
    """F6: vol and relative spread vs tau for N = 250 and N = 1000 with CIs (H3)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    styles = {250: (ASK_C, "N = 250 (thin)"), 1000: (TYPE_COLORS["fundamentalist"], "N = 1000")}
    taus = sorted(cm["tau"].unique())
    for ax, m in zip(axes, ("vol", "rel_spread_mean"), strict=True):
        for n_agents, (color, label) in styles.items():
            sub = cm[(cm["metric"] == m) & (cm["N"] == n_agents)].sort_values("tau")
            x = _tau_pos(taus, sub["tau"])
            ax.plot(x, sub["mean"], "o-", color=color, ms=4, label=label)
            ax.fill_between(x, sub["ci_low"], sub["ci_high"], color=color, alpha=0.2,
                            lw=0)  # fmt: skip
        _tau_axis(ax, taus)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
    fig.suptitle("F6. Tax effect in thin vs thick markets (mean, 95% CI; M = 10 in both)",
                 fontsize=11)  # fmt: skip
    return fig


def fig_sobol(idx: pd.DataFrame) -> Figure:
    """F8: Sobol first-order (S1) and total (ST) indices with 95% CIs per output."""
    outputs = list(dict.fromkeys(idx["output"]))
    fig, axes = plt.subplots(1, len(outputs), figsize=(3.8 * len(outputs), 3.6), sharey=True,
                             layout="constrained")  # fmt: skip
    for ax, out in zip(np.atleast_1d(axes), outputs, strict=True):
        sub = idx[idx["output"] == out]
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["S1"], 0.36, yerr=sub["S1_conf"], color=MUTED, capsize=3,
               label="S1 (first order)")  # fmt: skip
        ax.bar(x + 0.18, sub["ST"], 0.36, yerr=sub["ST_conf"], color=INK_2, capsize=3,
               label="ST (total)")  # fmt: skip
        ax.set_xticks(x, sub["param"])
        ax.set_title(f"PH cut effect on {out}", fontsize=9)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Sobol index")
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.suptitle("F8. What drives the PH cut effect: Sobol indices (95% CI)", fontsize=11)
    return fig


def fig_switching(shares: pd.DataFrame, runs: pd.DataFrame, T_burn: int) -> Figure:  # noqa: N803
    """F7: chartist share over time under switching, mean and 95% CI over seeds, per tau.

    ``shares``: long table (run_id, t, fundamentalist, chartist, noise) from E3;
    ``runs``: the runs table (run_id, tau, beta, ...).
    """
    df = shares.merge(runs[["run_id", "tau", "beta"]], on="run_id")
    betas = sorted(df["beta"].unique())
    taus = sorted(df["tau"].unique())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(betas), figsize=(6 * len(betas), 3.8), sharey=True,
                             layout="constrained")  # fmt: skip
    for ax, b in zip(np.atleast_1d(axes), betas, strict=True):
        for j, tau in enumerate(taus):
            g = df[(df["beta"] == b) & np.isclose(df["tau"], tau)].groupby("t")["chartist"]
            mean, sd, n = g.mean(), g.std(ddof=1), g.count()
            half = 1.96 * sd / np.sqrt(n)
            color = cmap(j / max(1, len(taus) - 1) * 0.9)
            bold = np.isclose(tau, PH_OLD) or np.isclose(tau, PH_NEW)
            ax.plot(mean.index, mean, color=color, lw=1.8 if bold else 1.1,
                    label=f"τ = {pct(tau)}" + (" (PH)" if bold else ""))  # fmt: skip
            ax.fill_between(mean.index, mean - half, mean + half, color=color, alpha=0.15, lw=0)
        _shade_burn(ax, T_burn)
        ax.set(xlabel="step", title=f"β = {b:g}")
        _style(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("chartist share of all agents")
    first.legend(fontsize=8, frameon=False)
    fig.suptitle("F7. Chartist share under strategy switching, by tax rate "
                 "(mean over seeds, 95% CI; burn-in shaded)", fontsize=11)  # fmt: skip
    return fig
