"""Price cleaning: validation, gap filling, proxy backfill, alignment and returns.

Frames have a ``DatetimeIndex`` named ``"date"`` and one column per ticker
(column index named ``"ticker"``). Docstrings state whether an input is
prices or returns, and no function modifies its input in place.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .. import TRADING_DAYS


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
# Full cleaning pipeline
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
    from ..data.sources import download_prices

    raw = download_prices(tickers, start=start, end=end)
    return build_panel(raw, **kwargs)


def stress_panel(cfg: dict, raw_prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    """Long-history holding prices for stress replays, proxy-backfilled where configured.

    Returns the price panel and each backfilled holding's first real date.
    """
    holdings = cfg["holdings"]
    full = fill_gaps(raw_prices[list(holdings)].sort_index(), cfg["panel"]["max_gap"])
    real_from: dict[str, pd.Timestamp] = {}
    for t, spec in holdings.items():
        p = spec.get("proxy") if isinstance(spec, Mapping) else None
        if p and p in raw_prices.columns:
            try:
                full, first = backfill_with_proxy(full, t, raw_prices[p].sort_index().ffill(limit=cfg["panel"]["max_gap"]))
            except ValueError:
                continue
            if first is not None:
                real_from[t] = first
    return full, real_from
