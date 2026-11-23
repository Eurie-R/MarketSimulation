"""Download daily closes for the empirical target and save cleaned log returns (Section 12.1).

Usage: uv run python scripts/fetch_psei.py [--ticker EPHE] [--start 2015-01-01]

Order of preference:
1. A user-supplied PSEi CSV at data/raw/psei.csv (columns Date,Close), if present.
2. Otherwise a Yahoo Finance download of ``--ticker`` (default EPHE).

Why EPHE: Yahoo has no history for PSEI.PS (only the current day's quote). We use
EPHE, the iShares MSCI Philippines ETF, as a proxy. It holds Philippine large
caps like the PSEi, but it is priced in USD and trades in New York, so its returns also carry
the PHP/USD exchange rate and US-session timing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from marketabm.empirical import (
    PROCESSED,
    RAW_CSV,
    clean_returns,
    load_raw_csv,
    save_returns,
    save_source,
)

DEFAULT_TICKER = "EPHE"
# ASSUMPTION: a download with fewer than 250 rows (about one trading year) counts as failed.
MIN_ROWS = 250


def download(ticker: str, start: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        return None
    if df is None or len(df) < MIN_ROWS:
        print(f"Download of {ticker} returned {0 if df is None else len(df)} rows.")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()[["Date", "Close", "Volume"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=DEFAULT_TICKER)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    if RAW_CSV.exists():
        print(f"Using user-supplied PSEi data at {RAW_CSV}")
        df, source = load_raw_csv(RAW_CSV), "PSEi (user CSV)"
    else:
        df = download(args.ticker, args.start)
        if df is None:
            print(
                "\nACTION NEEDED: place a CSV with columns Date,Close (PSEi daily closes from "
                f"{args.start} to today) at {RAW_CSV.as_posix()}, or pass another --ticker."
            )
            return 1
        raw = RAW_CSV.parent / f"{args.ticker.lower().lstrip('^')}.csv"
        raw.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw, index=False)
        print(f"Saved raw data to {raw} ({len(df)} rows)")
        source = f"{args.ticker} (Yahoo Finance)"

    r = clean_returns(df)
    if len(r) < MIN_ROWS:
        print(f"ERROR: only {len(r)} returns after cleaning; nothing saved.")
        return 1
    path = save_returns(r, Path(PROCESSED))
    save_source(source, r)
    print(
        f"Saved {len(r)} daily log returns from {source} ({r.index.min():%Y-%m-%d} to "
        f"{r.index.max():%Y-%m-%d}) to {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
