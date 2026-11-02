"""Price data: download, validation, cleaning, alignment and returns.

Frames have a ``DatetimeIndex`` named ``"date"`` and one column per ticker
(column index named ``"ticker"``). Docstrings state whether an input is
prices or returns, and no function modifies its input in place.

:func:`download_prices` is the only function in the library that uses the
network (Yahoo Finance through ``yfinance``); :func:`load_prices_csv` reads
any CSV with a ``date`` column and one numeric column per ticker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Download / load
# ---------------------------------------------------------------------------
def download_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily close PRICES from Yahoo Finance.

    Parameters
    ----------
    tickers : sequence of str
        Ticker symbols, e.g. ``["AAPL", "MSFT"]``.
    start, end : str
        ISO dates, e.g. ``"2015-01-01"``. ``end=None`` means today.

    Returns
    -------
    pd.DataFrame
        PRICES, one column per ticker in the order given. Adjusted for
        splits and dividends (``auto_adjust=True``).

    Notes
    -----
    Requires network access and the ``yfinance`` package. The import is lazy
    so the rest of the library works offline.
    """
    import yfinance as yf  # imported lazily so the library works offline

    tickers = list(dict.fromkeys(tickers))  # drop duplicates, keep order
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"yfinance returned no data for {tickers} from {start} to {end}"
        )
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(f"yfinance returned no columns for: {missing}")
    prices = prices[tickers].astype(float)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    prices.columns.name = "ticker"
    return prices


def save_prices_csv(prices: pd.DataFrame, path: str | Path) -> None:
    """Save a PRICES frame to CSV (date index, one column per ticker)."""
    prices.to_csv(path, index_label="date")


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load a PRICES frame from a CSV with a ``date`` column and one column per ticker."""
    prices = pd.read_csv(path, index_col="date", parse_dates=True)
    prices = prices.astype(float)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices.columns.name = "ticker"
    return prices


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Check that a PRICES frame is usable and return a sorted copy.

    Raises
    ------
    TypeError
        If the index is not a ``DatetimeIndex``.
    ValueError
        If there are duplicate dates or any price is <= 0.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex (one row per date)")
    if prices.index.has_duplicates:
        raise ValueError("prices index contains duplicate dates")
    if bool((prices <= 0).any().any()):
        raise ValueError("prices must be strictly positive (log returns need P > 0)")
    prices = prices.sort_index()
    prices.index.name = "date"
    prices.columns.name = "ticker"
    return prices


def fill_gaps(prices: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Forward-fill short runs of missing PRICES.

    Carries the last known price forward for at most ``max_gap`` consecutive
    missing days. Never fills backwards: missing values before a ticker's
    first price stay missing (no look-ahead).
    """
    return prices.ffill(limit=max_gap)


def backfill_with_proxy(
    prices: pd.DataFrame, ticker: str, proxy: pd.Series
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Extend ``ticker``'s PRICES backwards using ``proxy``'s returns.

    Before the ticker's first real price, its price path is reconstructed
    so that its log returns equal the proxy's log returns (chained backwards
    from the first real price). Real data is never altered. This is the
    usual treatment for a recent IPO or spin-off (e.g. a spun-off company
    proxied by its former parent).

    Parameters
    ----------
    prices : pd.DataFrame
        PRICES containing ``ticker`` (with leading NaNs).
    proxy : pd.Series
        Proxy PRICES on the same (or wider) date index.

    Returns
    -------
    (prices, first_real_date)
        A copy with the backfilled column, and the date from which the data
        is real (``None`` if nothing was backfilled).
    """
    out = prices.copy()
    col = out[ticker]
    first = col.first_valid_index()
    if first is None:
        raise ValueError(f"{ticker} has no prices at all; cannot backfill")
    before = out.index[out.index < first]
    if len(before) == 0:
        return out, None
    p = proxy.reindex(out.index).ffill()
    if p.loc[first : first].isna().any():
        raise ValueError(f"proxy has no price on {first.date()} for {ticker}")
    # P_t = P_first * proxy_t / proxy_first  for t < first
    scaled = float(col[first]) * p.loc[before] / float(p[first])
    out.loc[before, ticker] = scaled.values
    return out, first


def drop_short_history(prices: pd.DataFrame, min_obs: int = TRADING_DAYS) -> pd.DataFrame:
    """Keep only tickers with at least ``min_obs`` non-missing PRICES."""
    counts = prices.notna().sum()
    keep = counts[counts >= min_obs].index
    return prices[keep]


def common_window(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates on which every ticker has a PRICE (rectangular panel)."""
    return prices.dropna(how="any")


# ---------------------------------------------------------------------------
# Returns, weights and coverage
# ---------------------------------------------------------------------------
def to_returns(
    prices: pd.DataFrame, kind: Literal["log", "simple"] = "log"
) -> pd.DataFrame:
    """Convert PRICES to RETURNS.

    Parameters
    ----------
    prices : pd.DataFrame
        PRICES.
    kind : {"log", "simple"}
        ``"log"`` gives ln(P_t / P_{t-1}); ``"simple"`` gives P_t / P_{t-1} - 1.

    Returns
    -------
    pd.DataFrame
        RETURNS, with the all-missing first row removed. A return is missing
        wherever either of its two prices is missing (gaps are not bridged).
    """
    if kind == "log":
        returns = np.log(prices / prices.shift(1))
    elif kind == "simple":
        returns = prices / prices.shift(1) - 1.0
    else:
        raise ValueError(f"kind must be 'log' or 'simple', got {kind!r}")
    return returns.dropna(how="all")


def log_to_simple(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convert log RETURNS to simple RETURNS: exp(r) - 1."""
    return np.exp(returns) - 1.0


def simple_to_log(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convert simple RETURNS to log RETURNS: ln(1 + R)."""
    return np.log1p(returns)


def normalize_weights(
    weights: Mapping[str, float], tickers: Sequence[str]
) -> pd.Series:
    """Turn raw weights into a long-only weight vector summing to 1.

    Parameters
    ----------
    weights : mapping of str to float
        Raw weights (e.g. dollar amounts). Extra tickers not in ``tickers``
        are ignored.
    tickers : sequence of str
        The tickers in the panel; the output follows this order.

    Returns
    -------
    pd.Series
        Named ``"weight"``, indexed by ticker, summing to 1.

    Raises
    ------
    KeyError
        If a ticker in ``tickers`` has no weight.
    ValueError
        If any weight is negative, or the weights sum to zero or less.
    """
    w = pd.Series(dict(weights), dtype=float)
    missing = set(tickers) - set(w.index)
    if missing:
        raise KeyError(f"no weight given for: {sorted(missing)}")
    w = w.reindex(list(tickers))
    if bool((w < 0).any()):
        raise ValueError("weights must be non-negative (long-only)")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    w = w / total
    w.name = "weight"
    w.index.name = "ticker"
    return w


def coverage_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """Summarize data coverage per ticker for a PRICES frame.

    Returns
    -------
    pd.DataFrame
        Indexed by ticker, with columns ``first_date``, ``last_date``,
        ``n_obs`` and ``pct_missing`` (0-100).
    """
    summary = pd.DataFrame(
        {
            "first_date": prices.apply(pd.Series.first_valid_index),
            "last_date": prices.apply(pd.Series.last_valid_index),
            "n_obs": prices.notna().sum(),
            "pct_missing": prices.isna().mean() * 100.0,
        }
    )
    summary.index.name = "ticker"
    return summary


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_panel(
    prices: pd.DataFrame,
    max_gap: int = 5,
    min_obs: int = TRADING_DAYS,
    common: bool = True,
    kind: Literal["log", "simple"] = "log",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline on raw PRICES.

    Order matters: fill gaps before dropping short histories (so a few gaps
    do not push a ticker under ``min_obs``), and drop short histories before
    taking the common window (so one late IPO does not truncate everyone).

    Returns
    -------
    (prices, returns) : tuple of pd.DataFrame
        Clean PRICES and the matching RETURNS.
    """
    prices = validate_prices(prices)
    prices = fill_gaps(prices, max_gap=max_gap)
    prices = drop_short_history(prices, min_obs=min_obs)
    if prices.shape[1] == 0:
        raise ValueError(f"no ticker has at least {min_obs} observations")
    if common:
        prices = common_window(prices)
    returns = to_returns(prices, kind=kind)
    return prices, returns


def load_panel(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download PRICES and run :func:`build_panel` on them (network)."""
    raw = download_prices(tickers, start=start, end=end)
    return build_panel(raw, **kwargs)
