"""Price sources: Yahoo Finance, CSV files and sector labels.

:func:`download_prices` and :func:`fetch_sectors` are the only functions in
the library that use the network (Yahoo Finance through ``yfinance``);
:func:`load_prices_csv` reads any CSV with a ``date`` column and one numeric
column per ticker. Frames have a ``DatetimeIndex`` named ``"date"`` and one
column per ticker (column index named ``"ticker"``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ..config import data_start
from ..models.factors import factor_tickers
from ..preprocessing.portfolio import holdings_sectors


# ---------------------------------------------------------------------------
# download / load
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
# pipeline inputs
# ---------------------------------------------------------------------------
def fetch_data(cfg: dict, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PRICES for holdings and factor proxies from :func:`data_start`, from yfinance or CSV."""
    tickers = list(cfg["holdings"].keys())
    proxies = [s["proxy"] for s in cfg["holdings"].values() if isinstance(s, Mapping) and s.get("proxy")]
    for p in proxies:
        if p not in tickers:  # a shared proxy is added once
            tickers.append(p)
    ftick = factor_tickers(cfg["factors"])
    src = cfg["data"]["source"]
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if src == "yfinance":
        first = data_start(cfg)
        prices = download_prices(tickers, first, cfg["end"])
        fprices = download_prices(ftick, first, cfg["end"])
        save_prices_csv(prices, data_dir / "prices.csv")
        save_prices_csv(fprices, data_dir / "factor_prices.csv")
    elif src == "csv":
        prices = load_prices_csv(cfg["data"]["prices_csv"] or data_dir / "prices.csv")
        fprices = load_prices_csv(cfg["data"]["factor_csv"] or data_dir / "factor_prices.csv")
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise KeyError(f"{missing} not in {cfg['data']['prices_csv'] or data_dir / 'prices.csv'}; re-run without --offline")
        prices = prices[tickers]
    else:
        raise ValueError("data.source must be 'yfinance' or 'csv'")
    return prices, fprices


def fetch_sectors(cfg: dict, tickers: list[str], out_dir: Path) -> dict[str, str]:
    """Sector per ticker: YAML ``sector:`` first, then a cached yfinance lookup."""
    sectors = holdings_sectors(cfg["holdings"])
    cache = out_dir / "data" / "sectors.json"
    cached: dict[str, str] = {}
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
    need = [t for t in tickers if t not in sectors and t not in cached]
    if need and cfg["data"]["source"] == "yfinance" and cfg["data"].get("lookup_sectors", True):
        try:
            import yfinance as yf

            for t in need:
                try:
                    info = yf.Ticker(t).info or {}
                    cached[t] = str(info.get("sector") or "Unknown")
                except Exception:
                    cached[t] = "Unknown"
        except ImportError:
            pass
    out = {t: sectors.get(t, cached.get(t, "Unknown")) for t in tickers}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({**cached, **sectors}, indent=2), encoding="utf-8")
    return out
