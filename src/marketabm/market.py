"""Main simulation loop: ``run(config, seed) -> SimulationResult`` (CLAUDE.md Section 7).

``live(config, seed)`` runs the same loop and yields snapshots for the live view.
"""

from __future__ import annotations

import functools
import logging
import math
import platform
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any

import numpy as np

from marketabm.agents import SKIP_TAX, Agents, decide, init_agents
from marketabm.config import ModelConfig, VolFeedback, config_hash
from marketabm.orderbook import Fill, OrderBook
from marketabm.results import SimulationResult
from marketabm.rng import make_streams
from marketabm.switching import switching_round

log = logging.getLogger(__name__)

NAN = float("nan")


# --------------------------------------------------------------------------- pre-draws
def fundamental_path(rng: np.random.Generator, T: int, p0: float, sigma_f: float) -> np.ndarray:  # noqa: N803
    """Log-random-walk fundamental ``f_0 .. f_{T-1}`` with ``f_0 = p0`` (Section 5.3)."""
    z = rng.standard_normal(T - 1)
    logf = np.concatenate(([math.log(p0)], math.log(p0) + np.cumsum(sigma_f * z)))
    return np.exp(logf)


def draw_activation(rng: np.random.Generator, T: int, M: int, N: int) -> np.ndarray:  # noqa: N803
    """(T, M) array: M distinct agent ids per step, uniformly without replacement."""
    out = np.empty((T, M), dtype=np.int64)
    for t in range(T):
        out[t] = rng.choice(N, size=M, replace=False)
    return out


def draw_noise(rng: np.random.Generator, T: int, M: int, sigma_eps: float) -> np.ndarray:  # noqa: N803
    """(T, M) array of noise draws; slot j at step t always uses ``noise[t, j]``."""
    return rng.normal(0.0, sigma_eps, size=(T, M))


# --------------------------------------------------------------------------- ledger
class Ledger:
    """Cash, positions and tax revenue (Section 5.6). Tax is charged to the seller."""

    def __init__(self, N: int, tau: float) -> None:  # noqa: N803
        self.cash = [0.0] * N
        self.pos = [0] * N
        self.tau = tau
        self.tax_revenue = 0.0

    def apply_fill(self, buyer: int, seller: int, price: float, qty: int) -> None:
        value = price * qty
        self.cash[buyer] -= value
        self.pos[buyer] += qty
        self.cash[seller] += value * (1.0 - self.tau)
        self.pos[seller] -= qty
        self.tax_revenue += self.tau * value

    def wealth(self, p: float) -> np.ndarray:
        return np.asarray(self.cash) + np.asarray(self.pos, dtype=float) * p


# --------------------------------------------------------------------------- vol feedback
class VolFeedbackState:
    """Noise multiplier ``m_t`` from fast/slow EWMA volatility (Section 5.10).

    ``m_t = clip((s_fast / s_slow) ** gamma, 1 / cap, cap)`` where ``s_fast``, ``s_slow`` are
    bias-corrected EWMA standard deviations of one-step log returns up to step t - 1.
    Disabled: ``mult`` stays 1.0. Uses no random numbers.
    """

    def __init__(self, cfg: VolFeedback) -> None:
        self.enabled = cfg.enabled
        self.gamma, self.cap = cfg.gamma, cfg.cap
        self.lf = 0.5 ** (1.0 / cfg.fast_halflife)
        self.ls = 0.5 ** (1.0 / cfg.slow_halflife)
        self.ef = self.es = 0.0  # EWMA of r^2
        self.wf = self.ws = 0.0  # EWMA of 1 (bias correction)
        self.mult = 1.0

    def update(self, r: float) -> None:
        if not self.enabled:
            return
        r2 = r * r
        lf, ls = self.lf, self.ls
        self.ef = lf * self.ef + (1.0 - lf) * r2
        self.wf = lf * self.wf + (1.0 - lf)
        self.es = ls * self.es + (1.0 - ls) * r2
        self.ws = ls * self.ws + (1.0 - ls)
        var_slow = self.es / self.ws
        if var_slow <= 0.0:
            self.mult = 1.0
            return
        ratio = ((self.ef / self.wf) / var_slow) ** (0.5 * self.gamma)
        self.mult = min(self.cap, max(1.0 / self.cap, ratio))


# --------------------------------------------------------------------------- loop
class _Market:
    """Mutable simulation state. Only this module holds mutable state."""

    def __init__(self, cfg: ModelConfig, agents: Agents, f: np.ndarray) -> None:
        self.cfg = cfg
        self.agents = agents
        self.book = OrderBook(cfg.tick)
        self.ledger = Ledger(cfg.N, cfg.tau)
        self.logf = np.log(f).tolist()
        self.logp0 = math.log(cfg.p0)
        self.p_ticks = round(cfg.p0 / cfg.tick)
        self.p = cfg.p0
        self.logp = self.logp0
        self.logP = [0.0] * cfg.T
        self.agent_order = [-1] * cfg.N
        self.size_unit = cfg.size_unit if cfg.order_size == "signal" else None
        self._refresh_params()
        # step accumulators (reset every step)
        self.s_fills = [0, 0, 0]
        self.s_aggr = [0, 0, 0]
        self.s_skip_tax = [0, 0, 0]
        self.s_skip_none = [0, 0, 0]
        self.s_active = [0, 0, 0]
        self.s_volume = 0

    def _refresh_params(self) -> None:
        a = self.agents
        self.types = a.types.tolist()
        self.g1, self.g2, self.n = a.g1.tolist(), a.g2.tolist(), a.n.tolist()
        self.L, self.h, self.k = a.L.tolist(), a.h.tolist(), a.k.tolist()

    def expire(self, t: int) -> None:
        ao = self.agent_order
        for o in self.book.expire_orders(t):
            if ao[o.agent_id] == o.order_id:
                ao[o.agent_id] = -1

    def activate(self, t: int, i: int, eps: float) -> None:
        cfg = self.cfg
        typ = self.types[i]
        self.s_active[typ] += 1
        L = self.L[i]  # noqa: N806
        lag = t - L
        logp_lag = self.logP[lag] if lag >= 0 else self.logp0
        side, price_ticks, qty = decide(
            self.logp, self.logf[t], logp_lag, L, self.g1[i], self.g2[i], self.n[i],
            self.h[i], self.k[i], eps, cfg.tau_f, cfg.tau, cfg.c_min, cfg.tick,
            cfg.price_band, self.size_unit, cfg.size_max,
        )  # fmt: skip
        if side > 1:
            if side == SKIP_TAX:
                self.s_skip_tax[typ] += 1
            else:
                self.s_skip_none[typ] += 1
            return
        book = self.book
        ao = self.agent_order
        if ao[i] >= 0:
            book.cancel(ao[i])
        resting_id, fills = book.submit(i, side, price_ticks, qty, t, t + cfg.order_life)
        if fills:
            self._apply_fills(fills, typ)
        ao[i] = -1 if resting_id is None else resting_id

    def _apply_fills(self, fills: list[Fill], aggressor_type: int) -> None:
        types, ao, book, ledger = self.types, self.agent_order, self.book, self.ledger
        tick = self.cfg.tick
        for fl in fills:
            price = fl.price_ticks * tick
            ledger.apply_fill(fl.buyer_id, fl.seller_id, price, fl.qty)
            self.s_fills[types[fl.buyer_id]] += 1
            self.s_fills[types[fl.seller_id]] += 1
            self.s_aggr[aggressor_type] += 1
            self.s_volume += fl.qty
            passive = fl.seller_id if fl.aggressor_side == 1 else fl.buyer_id
            if ao[passive] >= 0 and not book.is_live(ao[passive]):
                ao[passive] = -1
        last = fills[-1]
        self.p_ticks = last.price_ticks
        self.p = last.price_ticks * tick
        self.logp = math.log(self.p)

    def reset_step(self) -> None:
        for arr in (self.s_fills, self.s_aggr, self.s_skip_tax, self.s_skip_none, self.s_active):
            arr[0] = arr[1] = arr[2] = 0
        self.s_volume = 0

    def check_invariants(self) -> None:
        """Debug-mode checks: money and units conserved; at most one live order per agent."""
        led = self.ledger
        total_cash = math.fsum(led.cash) + led.tax_revenue
        scale = max(1.0, math.fsum(abs(c) for c in led.cash))
        assert abs(total_cash) <= 1e-9 * scale, f"money not conserved: {total_cash}"
        assert sum(led.pos) == 0, "positions do not net to zero"
        per_agent = np.bincount(
            [o.agent_id for o in self.book.orders.values()], minlength=self.cfg.N
        )
        assert per_agent.max(initial=0) <= 1, "agent with more than one live order"


class _Recorder:
    """Per-step and snapshot recorders."""

    def __init__(self, cfg: ModelConfig) -> None:
        T = cfg.T  # noqa: N806
        self.price = np.empty(T)
        self.best_bid = np.full(T, NAN)
        self.best_ask = np.full(T, NAN)
        self.volume = np.zeros(T, dtype=np.int64)
        self.tax_cum = np.zeros(T)
        self.vol_mult = np.ones(T)
        self.rows: list[tuple[list[int], ...]] = []
        self.snap_t: list[int] = []
        self.depth_bid: list[float] = []
        self.depth_ask: list[float] = []
        self.book_size: list[int] = []
        self.type_shares: list[np.ndarray] = []
        self.switch_t: list[int] = []
        self.switch_pi: list[np.ndarray] = []

    def record_step(self, t: int, m: _Market) -> None:
        self.price[t] = m.p
        bb, ba = m.book.best_bid(), m.book.best_ask()
        tick = m.cfg.tick
        if bb is not None:
            self.best_bid[t] = bb * tick
        if ba is not None:
            self.best_ask[t] = ba * tick
        self.volume[t] = m.s_volume
        self.tax_cum[t] = m.ledger.tax_revenue
        self.rows.append(
            (m.s_fills[:], m.s_aggr[:], m.s_skip_tax[:], m.s_skip_none[:], m.s_active[:])
        )

    def record_snapshot(self, t: int, m: _Market) -> None:
        bb, ba = m.book.best_bid(), m.book.best_ask()
        self.snap_t.append(t)
        self.book_size.append(len(m.book))
        if bb is None or ba is None:
            self.depth_bid.append(NAN)
            self.depth_ask.append(NAN)
            return
        db, da = m.book.depth_within((bb + ba) / 2.0, m.cfg.depth_pct)
        self.depth_bid.append(float(db))
        self.depth_ask.append(float(da))


# --------------------------------------------------------------------------- run
def run(config: ModelConfig, seed: int, *, debug: bool = False) -> SimulationResult:
    """Run one simulation.

    Parameters
    ----------
    config : ModelConfig
        Fully resolved model configuration.
    seed : int
        Root seed for all random streams (Section 8).
    debug : bool
        If True, check conservation invariants after every step (slow).

    Returns
    -------
    SimulationResult
    """
    t0 = time.perf_counter()
    sim = Simulation(config, seed)
    for t in range(config.T):
        sim.step(t, debug)
    return sim.result(time.perf_counter() - t0)


class Simulation:
    """One simulation: pre-draws (Section 8) plus the per-step loop body (Section 7).

    ``run`` drives it in one go; the live view drives it with :meth:`advance` and reads
    :meth:`frame`. Both use the same loop and streams, so the paths are identical.
    """

    def __init__(self, cfg: ModelConfig, seed: int) -> None:
        self.cfg, self.seed = cfg, seed
        self.t = 0  # next step to run
        self.streams = make_streams(seed)
        self.f = fundamental_path(self.streams.fundamental, cfg.T, cfg.p0, cfg.sigma_f)
        agents = init_agents(cfg, self.streams.init)
        keys = ("types", "g1", "g2", "n", "L", "h", "k")
        self.params0 = {k: getattr(agents, k).copy() for k in keys}
        self.activation = draw_activation(self.streams.activation, cfg.T, cfg.M, cfg.N).tolist()
        self.noise = draw_noise(self.streams.noise, cfg.T, cfg.M, cfg.sigma_eps).tolist()
        self.m = _Market(cfg, agents, self.f)
        self.rec = _Recorder(cfg)
        self.wealth_prev = np.zeros(cfg.N)
        self.vf = VolFeedbackState(cfg.vol_feedback)

    @property
    def done(self) -> bool:
        return self.t >= self.cfg.T

    def step(self, t: int, debug: bool = False) -> None:
        cfg, m, rec = self.cfg, self.m, self.rec
        m.expire(t)
        m.reset_step()
        eps_row = self.noise[t]
        mult = self.vf.mult  # from returns up to t - 1; 1.0 when feedback is off
        for j, i in enumerate(self.activation[t]):
            m.activate(t, i, eps_row[j] * mult)
        m.logP[t] = m.logp
        self.vf.update(m.logp - (m.logP[t - 1] if t > 0 else m.logp0))
        rec.vol_mult[t] = mult
        rec.record_step(t, m)
        if (t + 1) % cfg.return_interval == 0:
            rec.record_snapshot(t, m)
        sw = cfg.switching
        if sw.enabled and (t + 1) % sw.S == 0:
            self.wealth_prev = _switch(m, rec, t, self.wealth_prev, self.streams.switching)
        if debug:
            m.check_invariants()
        self.t = t + 1

    def advance(self, n: int) -> None:
        """Run the next ``n`` steps (fewer if the run ends)."""
        for t in range(self.t, min(self.t + n, self.cfg.T)):
            self.step(t)

    def frame(self, window: int | None = None, levels: int = 25) -> LiveFrame:
        """Read-only snapshot of the last ``window`` completed steps (all if None)."""
        return _frame(self, window, levels)

    def result(self, runtime_s: float) -> SimulationResult:
        meta = run_metadata(self.cfg, self.seed)
        meta["runtime_s"] = runtime_s
        return _build_result(self.cfg, self.seed, meta, self.f, self.m, self.rec, self.params0)


@dataclass(frozen=True, slots=True)
class LiveFrame:
    """State of a running simulation after step ``t`` (for the live view; read-only copies).

    Per-step arrays cover steps ``t_start .. t``. ``bids`` / ``asks`` are ``(price, qty)``
    pairs of the aggregated book levels nearest the spread, best first.
    """

    t: int
    t_start: int
    price: np.ndarray
    fundamental: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    volume: np.ndarray
    fills_by_type: np.ndarray  # (steps, 3) participations per step
    skipped_tax_by_type: np.ndarray  # (steps, 3)
    active_by_type: np.ndarray  # (steps, 3)
    tax_revenue: float
    bids: list[tuple[float, int]]
    asks: list[tuple[float, int]]
    book_size: int


def live(
    config: ModelConfig, seed: int, every: int = 50, window: int | None = None, levels: int = 25
) -> Iterator[LiveFrame]:
    """Run a simulation and yield a :class:`LiveFrame` every ``every`` steps (and at the end)."""
    sim = Simulation(config, seed)
    while not sim.done:
        sim.advance(every)
        yield sim.frame(window, levels)


def _frame(sim: Simulation, window: int | None, levels: int) -> LiveFrame:
    rec, m, tick = sim.rec, sim.m, sim.cfg.tick
    n = sim.t
    a = 0 if window is None else max(0, n - window)
    rows = np.asarray(rec.rows[a:n], dtype=np.int64).reshape(n - a, 5, 3)
    bids, asks = m.book.levels(levels)
    return LiveFrame(
        t=n - 1,
        t_start=a,
        price=rec.price[a:n].copy(),
        fundamental=sim.f[a:n],
        best_bid=rec.best_bid[a:n].copy(),
        best_ask=rec.best_ask[a:n].copy(),
        volume=rec.volume[a:n].copy(),
        fills_by_type=rows[:, 0],
        skipped_tax_by_type=rows[:, 2],
        active_by_type=rows[:, 4],
        tax_revenue=m.ledger.tax_revenue,
        bids=[(p * tick, q) for p, q in bids],
        asks=[(p * tick, q) for p, q in asks],
        book_size=len(m.book),
    )


def _switch(
    m: _Market, rec: _Recorder, t: int, wealth_prev: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    cfg = m.cfg
    wealth_now = m.ledger.wealth(m.p)
    assert cfg.switching.beta is not None
    _, pi = switching_round(
        m.agents, wealth_now, wealth_prev, cfg.switching.beta, cfg.p0, cfg.profiles, rng
    )
    m._refresh_params()
    rec.type_shares.append(m.agents.counts() / cfg.N)
    rec.switch_t.append(t)
    rec.switch_pi.append(pi)
    return wealth_now


def _build_result(
    cfg: ModelConfig,
    seed: int,
    meta: dict[str, Any],
    f: np.ndarray,
    m: _Market,
    rec: _Recorder,
    params0: dict[str, np.ndarray],
) -> SimulationResult:
    cols = [np.array([row[c] for row in rec.rows], dtype=np.int64) for c in range(5)]
    return SimulationResult(
        config=cfg,
        seed=seed,
        metadata=meta,
        price=rec.price,
        fundamental=f,
        best_bid=rec.best_bid,
        best_ask=rec.best_ask,
        volume=rec.volume,
        tax_revenue_cum=rec.tax_cum,
        vol_mult=rec.vol_mult,
        fills_by_type=cols[0],
        aggressive_fills_by_type=cols[1],
        skipped_tax_by_type=cols[2],
        skipped_no_signal_by_type=cols[3],
        active_by_type=cols[4],
        snapshot_t=np.array(rec.snap_t, dtype=np.int64),
        depth_bid=np.array(rec.depth_bid),
        depth_ask=np.array(rec.depth_ask),
        book_size=np.array(rec.book_size, dtype=np.int64),
        type_shares=np.array(rec.type_shares).reshape(-1, 3),
        switch_t=np.array(rec.switch_t, dtype=np.int64),
        switch_pi=np.array(rec.switch_pi).reshape(-1, 2),
        types=m.agents.types.copy(),
        cash=np.array(m.ledger.cash),
        pos=np.array(m.ledger.pos, dtype=np.int64),
        agent_params=params0,
    )


# --------------------------------------------------------------------------- metadata
@functools.cache
def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def run_metadata(cfg: ModelConfig, seed: int) -> dict[str, Any]:
    """Config, seed, config hash, git commit, Python and package versions."""
    versions = {}
    for pkg in ("numpy", "pandas", "scipy", "statsmodels", "pydantic"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "config": cfg.canonical_dict(),
        "seed": seed,
        "config_hash": config_hash(cfg),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "versions": versions,
    }
