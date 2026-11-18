"""Empirical return data: loading, cleaning, moments and block bootstrap (CLAUDE.md Section 12).

The target series is the PSEi when a user CSV is supplied, otherwise a proxy downloaded by
``scripts/fetch_psei.py`` (default EPHE). ``data/processed/source.json`` records which one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from marketabm.stylized_facts import MOMENT_NAMES, moments

log = logging.getLogger(__name__)

RAW_CSV = Path("data/raw/psei.csv")
PROCESSED = Path("data/processed/psei_returns.parquet")
SOURCE = Path("data/processed/source.json")


def clean_returns(df: pd.DataFrame) -> pd.Series:
    """Daily log returns from a frame with ``Date`` and ``Close`` (and optionally ``Volume``).

    Drops missing closes, non-positive closes and zero-volume rows (holidays), then takes
    log differences of consecutive remaining closes.

    Parameters
    ----------
    df : DataFrame
        Columns ``Date``, ``Close`` and optionally ``Volume``.

    Returns
    -------
    Series
        Log returns indexed by date, named ``r``.
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.dropna(subset=["Close"]).sort_values("Date")
    d = d[d["Close"] > 0]
    if "Volume" in d.columns:
        # ASSUMPTION: rows with missing volume are kept; only an explicit 0 marks a holiday.
        d = d[~(d["Volume"] == 0)]
    d = d.drop_duplicates(subset="Date", keep="last")
    r = np.log(d["Close"].to_numpy(dtype=float))
    out = pd.Series(np.diff(r), index=d["Date"].iloc[1:].to_numpy(), name="r")
    return out[np.isfinite(out)]


def load_raw_csv(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """Read a user-supplied CSV with at least ``Date,Close`` columns."""
    df = pd.read_csv(path)
    missing = {"Date", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}; need Date,Close")
    return df


def save_returns(r: pd.Series, path: str | Path = PROCESSED) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r.to_frame().rename_axis("Date").to_parquet(path)
    return path


def save_source(name: str, r: pd.Series, path: str | Path = SOURCE) -> Path:
    """Record which series the processed returns came from."""
    path = Path(path)
    meta = {
        "source": name,
        "n_returns": len(r),
        "start": f"{r.index.min():%Y-%m-%d}",
        "end": f"{r.index.max():%Y-%m-%d}",
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_source(path: str | Path = SOURCE) -> dict[str, object]:
    """Source metadata of the processed returns (``{"source": "unknown"}`` if missing)."""
    path = Path(path)
    if not path.exists():
        return {"source": "unknown"}
    return json.loads(path.read_text(encoding="utf-8"))


def load_returns(path: str | Path = PROCESSED) -> np.ndarray:
    """Load processed PSEi daily log returns as a float array."""
    return pd.read_parquet(path)["r"].to_numpy(dtype=float)


def moment_vector(r: np.ndarray) -> np.ndarray:
    """Section 12.2 moments as an array in ``MOMENT_NAMES`` order."""
    m = moments(r)
    return np.array([m[k] for k in MOMENT_NAMES], dtype=float)


def block_bootstrap_moments(
    r: np.ndarray, rng: np.random.Generator, block: int = 20, n_boot: int = 1000
) -> np.ndarray:
    """Moving-block bootstrap of the moment vector.

    Each resample concatenates ``ceil(n / block)`` blocks of length ``block`` whose start
    points are drawn uniformly from ``0 .. n - block``, truncated to length n.

    Returns
    -------
    ndarray, shape (n_boot, len(MOMENT_NAMES))
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 2 * block:
        raise ValueError(f"series too short ({n}) for block length {block}")
    n_blocks = -(-n // block)
    offsets = np.arange(block)
    out = np.empty((n_boot, len(MOMENT_NAMES)))
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
        out[b] = moment_vector(r[idx])
    return out


def moment_weights(boot: np.ndarray) -> np.ndarray:
    """Diagonal weights ``1 / var`` of each bootstrapped moment (NaN draws ignored)."""
    var = np.nanvar(boot, axis=0, ddof=1)
    return 1.0 / var
