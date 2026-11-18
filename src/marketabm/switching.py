"""Brock-Hommes logit strategy switching (Section 5.8, E3 only)."""

from __future__ import annotations

import numpy as np

from marketabm.agents import CHART, FUND, NOISE, Agents, map_weights
from marketabm.config import Profiles


def type_profits(
    wealth_now: np.ndarray, wealth_prev: np.ndarray, types: np.ndarray, p0: float
) -> np.ndarray:
    """Mean per-agent wealth change over the window, scaled by p0, for (FUND, CHART).

    A type with no agents gets 0.
    """
    d = (wealth_now - wealth_prev) / p0
    out = np.zeros(2)
    for j, k in enumerate((FUND, CHART)):
        mask = types == k
        if mask.any():
            out[j] = d[mask].mean()
    return out


def logit_probs(pi: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softmax of ``beta * pi``."""
    z = beta * np.asarray(pi, dtype=float)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def switching_round(
    agents: Agents,
    wealth_now: np.ndarray,
    wealth_prev: np.ndarray,
    beta: float,
    p0: float,
    profiles: Profiles,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one switching round in place on ``agents``.

    Every non-noise agent redraws its type. Agents whose type changed redraw
    (g1, g2, n) from the new profile; L, h and k stay fixed. The number of
    draws from ``rng`` depends only on the number of non-noise agents.

    Returns
    -------
    (changed_mask, pi)
        Boolean mask of agents whose type changed, and the (FUND, CHART) profits.
    """
    pi = type_profits(wealth_now, wealth_prev, agents.types, p0)
    prob = logit_probs(pi, beta)
    idx = np.flatnonzero(agents.types != NOISE)
    u_type = rng.random(len(idx))
    u_w = rng.random((3, len(idx)))
    new_types = np.where(u_type < prob[0], FUND, CHART).astype(np.int8)
    changed = np.zeros(agents.N, dtype=bool)
    changed[idx] = new_types != agents.types[idx]
    agents.types[idx] = new_types
    ch = np.flatnonzero(changed)
    if len(ch):
        sel = np.isin(idx, ch)
        t_ch = agents.types[ch]
        agents.g1[ch] = map_weights(t_ch, u_w[0, sel], profiles, "g1")
        agents.g2[ch] = map_weights(t_ch, u_w[1, sel], profiles, "g2")
        agents.n[ch] = map_weights(t_ch, u_w[2, sel], profiles, "n")
    return changed, pi
