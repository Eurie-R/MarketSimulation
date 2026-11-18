"""Agent population as NumPy arrays, type profiles and the decision rule (Section 5.4-5.5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from marketabm.config import ModelConfig, Profiles

FUND = 0
CHART = 1
NOISE = 2
TYPE_NAMES: tuple[str, str, str] = ("fundamentalist", "chartist", "noise")

# Non-order outcomes of `decide` (orders use side +1 / -1).
SKIP_TAX = 2
SKIP_NO_SIGNAL = 3


@dataclass
class Agents:
    """Per-agent state. Arrays have length N."""

    types: np.ndarray  # int8 in {FUND, CHART, NOISE}
    g1: np.ndarray
    g2: np.ndarray
    n: np.ndarray
    L: np.ndarray  # int
    h: np.ndarray  # int
    k: np.ndarray

    @property
    def N(self) -> int:  # noqa: N802 - matches model notation
        return len(self.types)

    def counts(self) -> np.ndarray:
        """Number of agents of each type, shape (3,)."""
        return np.bincount(self.types, minlength=3)


def type_counts(N: int, shares: tuple[float, float, float]) -> tuple[int, int, int]:  # noqa: N803
    """Agent counts from mix shares; fundamentalists absorb rounding (Section 5.4)."""
    _, s_c, s_n = shares
    n_c = round(N * s_c)
    n_n = round(N * s_n)
    n_f = N - n_c - n_n
    if n_f < 0:
        raise ValueError("rounded chartist + noise counts exceed N")
    return n_f, n_c, n_n


def map_weights(types: np.ndarray, u: np.ndarray, profiles: Profiles, field: str) -> np.ndarray:
    """Map U(0,1) draws to each agent's type range for weight ``field`` ('g1', 'g2' or 'n')."""
    lo = np.empty(len(types))
    hi = np.empty(len(types))
    for idx, prof in enumerate(profiles.by_index()):
        a, b = getattr(prof, field)
        mask = types == idx
        lo[mask] = a
        hi[mask] = b
    return lo + u * (hi - lo)


def init_agents(cfg: ModelConfig, rng: np.random.Generator) -> Agents:
    """Assign types and draw parameters from the ``init`` stream.

    The number of draws depends only on N, never on the mix (CRN rule 2).
    """
    N = cfg.N  # noqa: N806
    n_f, n_c, n_n = type_counts(N, cfg.mix.as_tuple())
    types = np.array([FUND] * n_f + [CHART] * n_c + [NOISE] * n_n, dtype=np.int8)
    rng.shuffle(types)
    u_g1 = rng.random(N)
    u_g2 = rng.random(N)
    u_n = rng.random(N)
    L = rng.integers(cfg.L_range[0], cfg.L_range[1] + 1, size=N)  # noqa: N806
    h = rng.integers(cfg.h_range[0], cfg.h_range[1] + 1, size=N)
    k = rng.uniform(0.0, cfg.k_max, size=N)
    return Agents(
        types=types,
        g1=map_weights(types, u_g1, cfg.profiles, "g1"),
        g2=map_weights(types, u_g2, cfg.profiles, "g2"),
        n=map_weights(types, u_n, cfg.profiles, "n"),
        L=L,
        h=h,
        k=k,
    )


def decide(
    logp: float,
    logf: float,
    logp_lag: float,
    L: int,  # noqa: N803
    g1: float,
    g2: float,
    n: float,
    h: int,
    k: float,
    eps: float,
    tau_f: float,
    tau: float,
    c_min: float,
    tick: float,
    price_band: float | None = None,
    size_unit: float | None = None,
    size_max: int = 1,
) -> tuple[int, int, int]:
    """Apply the decision rule of Section 5.5 for one activated agent.

    Parameters
    ----------
    logp : float
        Log of the current (last trade) price.
    logf : float
        Log fundamental value at this step.
    logp_lag : float
        Log of the recorded end-of-step price L steps ago (log p0 if t - L < 0).
    L, g1, g2, n, h, k : agent parameters.
    eps : float
        Noise draw for this activation slot.
    tau_f, tau, c_min, tick : model parameters.
    price_band : float or None
        If set, the order's log distance from the current price is clipped to
        ``[-price_band, price_band]`` before the k offset. The forecast and the
        participation filter are not affected.
    size_unit : float or None
        If set (``order_size = "signal"``), quantity is
        ``min(size_max, max(1, ceil(expected_gain / size_unit)))``; otherwise 1.
    size_max : int
        Largest order size.

    Returns
    -------
    (outcome, price_ticks, qty)
        outcome is +1 (buy) / -1 (sell) with the limit price in ticks and the
        quantity, or ``SKIP_TAX`` / ``SKIP_NO_SIGNAL`` with price_ticks = qty = 0.
    """
    f_sig = (logf - logp) / tau_f
    c_sig = (logp - logp_lag) / L
    r_hat = (g1 * f_sig + g2 * c_sig + n * eps) / (g1 + g2 + n)
    move = r_hat * h  # expected log price change, log(p_hat / p)
    gain = abs(move)
    if not gain > tau + c_min:
        return (SKIP_TAX if gain > c_min else SKIP_NO_SIGNAL), 0, 0
    if price_band is not None:
        move = max(-price_band, min(price_band, move))
    side = 1 if r_hat > 0 else -1
    price = math.exp(logp + move) * (1.0 - side * k)
    qty = 1 if size_unit is None else min(size_max, max(1, math.ceil(gain / size_unit)))
    return side, max(1, round(price / tick)), qty
