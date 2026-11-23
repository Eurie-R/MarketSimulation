"""SimulationResult container, summary metrics and serialization."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marketabm import metrics
from marketabm.agents import TYPE_NAMES
from marketabm.config import ModelConfig

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    config: ModelConfig
    seed: int
    metadata: dict[str, Any]
    # per-step (length T)
    price: np.ndarray
    fundamental: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    volume: np.ndarray
    tax_revenue_cum: np.ndarray
    vol_mult: np.ndarray  # noise multiplier m_t; all 1.0 when off
    fills_by_type: np.ndarray  # (T, 3)
    aggressive_fills_by_type: np.ndarray
    skipped_tax_by_type: np.ndarray
    skipped_no_signal_by_type: np.ndarray
    active_by_type: np.ndarray
    # snapshots (length T // return_interval)
    snapshot_t: np.ndarray
    depth_bid: np.ndarray
    depth_ask: np.ndarray
    book_size: np.ndarray
    # switching
    type_shares: np.ndarray  # (rounds, 3)
    switch_t: np.ndarray  # step index of each switching round
    switch_pi: np.ndarray  # (rounds, 2): profits pi_k of (FUND, CHART) used in each round
    # final agent state
    types: np.ndarray
    cash: np.ndarray
    pos: np.ndarray
    agent_params: dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def period_returns(self) -> np.ndarray:
        c = self.config
        return metrics.period_returns(self.price, c.T_burn, c.return_interval)

    def wealth(self) -> np.ndarray:
        """Final mark-to-market wealth per agent."""
        return self.cash + self.pos * self.price[-1]

    def summary(self) -> dict[str, float]:
        """Flat dict of all summary metrics after burn-in. Never raises."""
        out: dict[str, float] = {}
        for name, fn in (
            ("returns", self._returns_part),
            ("liquidity", self._liquidity_part),
            ("composition", self._composition_part),
            ("switching", self._switching_part),
        ):
            try:
                out.update(fn())
            except Exception:  # pragma: no cover - defensive: summary() never raises
                log.warning("summary part %r failed", name, exc_info=True)
        for key, val in out.items():
            if isinstance(val, float) and math.isnan(val):
                log.warning("metric %s is NaN (seed=%s)", key, self.seed)
        return out

    def _returns_part(self) -> dict[str, float]:
        b = self.config.T_burn
        out = metrics.return_metrics(self.period_returns())
        out["max_drawdown"] = metrics.max_drawdown(np.log(self.price[b:]))
        out["mispricing"] = metrics.mispricing(self.price[b:], self.fundamental[b:])
        return out

    def _liquidity_part(self) -> dict[str, float]:
        c = self.config
        b = c.T_burn
        snap = self.snapshot_t >= b
        rev_before = self.tax_revenue_cum[b - 1] if b >= 1 else 0.0
        rev = float(self.tax_revenue_cum[-1] - rev_before)
        return {
            "volume_total": float(self.volume[b:].sum()),
            "rel_spread_mean": metrics.rel_spread_mean(self.best_bid[b:], self.best_ask[b:]),
            "one_sided_share": metrics.one_sided_share(self.best_bid[b:], self.best_ask[b:]),
            "depth_mean": metrics.depth_mean(self.depth_bid[snap], self.depth_ask[snap]),
            "book_size_mean": (float(np.mean(self.book_size[snap])) if snap.any() else metrics.NAN),
            "tax_revenue": rev,
            "tax_revenue_per_step": rev / (c.T - b),
        }

    def _composition_part(self) -> dict[str, float]:
        b = self.config.T_burn
        # ASSUMPTION: with switching on, "number of agents of type k" uses final types.
        counts = np.bincount(self.types, minlength=3)
        return metrics.composition(
            self.fills_by_type[b:].sum(axis=0),
            self.aggressive_fills_by_type[b:].sum(axis=0),
            self.skipped_tax_by_type[b:].sum(axis=0),
            self.skipped_no_signal_by_type[b:].sum(axis=0),
            self.active_by_type[b:].sum(axis=0),
            counts,
            TYPE_NAMES,
        )

    def _switching_part(self) -> dict[str, float]:
        """Mean type shares over switching rounds after burn-in (E3). Empty if switching is off.

        ASSUMPTION: E3 needs a per-run summary of the mix, so this adds
        ``share_mean_<type>`` (mean over rounds with ``switch_t >= T_burn``) and
        ``share_sd_chartist`` (round-to-round spread, a measure of herding swings).
        """
        if len(self.switch_t) == 0:
            return {}
        sel = self.switch_t >= self.config.T_burn
        if not sel.any():
            return {}
        sh = self.type_shares[sel]
        out = {f"share_mean_{n}": float(sh[:, k].mean()) for k, n in enumerate(TYPE_NAMES)}
        out["share_sd_chartist"] = float(sh[:, 1].std(ddof=1)) if len(sh) > 1 else metrics.NAN
        return out

    # ------------------------------------------------------------- serialization
    def series_frame(self) -> pd.DataFrame:
        """Per-step series as a DataFrame (one row per step)."""
        data: dict[str, np.ndarray] = {
            "t": np.arange(len(self.price)),
            "price": self.price,
            "fundamental": self.fundamental,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "volume": self.volume,
            "tax_revenue_cum": self.tax_revenue_cum,
            "vol_mult": self.vol_mult,
        }
        for arr_name in (
            "fills_by_type",
            "aggressive_fills_by_type",
            "skipped_tax_by_type",
            "skipped_no_signal_by_type",
            "active_by_type",
        ):
            arr = getattr(self, arr_name)
            for k, tname in enumerate(TYPE_NAMES):
                data[f"{arr_name}_{tname}"] = arr[:, k]
        return pd.DataFrame(data)

    def save_series(self, directory: str | Path = "results/single") -> Path:
        """Write per-step series to ``<directory>/<run_id>.parquet``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        run_id = f"{self.metadata['config_hash'][:12]}_seed{self.seed}"
        path = directory / f"{run_id}.parquet"
        self.series_frame().to_parquet(path, index=False)
        return path
