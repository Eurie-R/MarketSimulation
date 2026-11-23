"""Named random streams built from one root seed."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

STREAM_NAMES: tuple[str, ...] = ("fundamental", "init", "activation", "noise", "switching")


class Streams(NamedTuple):
    """Independent generators; the spawn order is fixed and must never change."""

    fundamental: np.random.Generator
    init: np.random.Generator
    activation: np.random.Generator
    noise: np.random.Generator
    switching: np.random.Generator


def make_streams(seed: int) -> Streams:
    """Build the five named generators from ``seed``.

    Parameters
    ----------
    seed : int
        Root seed. The same seed always gives the same five streams.

    Returns
    -------
    Streams
        ``fundamental, init, activation, noise, switching`` in that order.
    """
    ss = np.random.SeedSequence(seed)
    return Streams(*(np.random.default_rng(s) for s in ss.spawn(len(STREAM_NAMES))))
