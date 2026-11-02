"""Holdings, portfolio return series and the :class:`Portfolio` object.

A :class:`Portfolio` holds long-only weights and the clean price/return panel
they apply to. Holdings may be given as dollars, shares or weights.

Two return conventions are available through ``rebalance``:

* ``"daily"``: constant weights (rebalanced every day). Used for all risk
  measures, which describe today's weights.
* ``"none"``: buy-and-hold, with weights drifting with prices.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..models.performance import performance_summary
from .panel import backfill_with_proxy, build_panel, log_to_simple, normalize_weights, simple_to_log


# ---------------------------------------------------------------------------
# Holdings -> weights
# ---------------------------------------------------------------------------
def holdings_value(holdings: Mapping[str, Any], last_prices: pd.Series) -> float | None:
    """Market value implied by the holdings, or None if any holding is a bare weight."""
    total = 0.0
    for ticker, spec in holdings.items():
        if isinstance(spec, Mapping):
            if "shares" in spec:
                total += float(spec["shares"]) * float(last_prices[ticker])
            elif "dollars" in spec:
                total += float(spec["dollars"])
            else:
                return None
        else:
            total += float(spec)
    return total


def holdings_sectors(holdings: Mapping[str, Any]) -> dict[str, str]:
    """Sector labels given in the holdings config (ticker -> sector)."""
    return {t: str(s["sector"]) for t, s in holdings.items() if isinstance(s, Mapping) and s.get("sector")}


def holdings_to_weights(
    holdings: Mapping[str, Any], last_prices: pd.Series | None = None
) -> pd.Series:
    """Convert a holdings mapping into normalized long-only weights.

    Each value is a number (dollars) or a dict with one of ``dollars``,
    ``shares`` or ``weight``. ``shares`` requires ``last_prices``.
    """
    dollars: dict[str, float] = {}
    for ticker, spec in holdings.items():
        if isinstance(spec, Mapping):
            if "dollars" in spec:
                dollars[ticker] = float(spec["dollars"])
            elif "weight" in spec:
                dollars[ticker] = float(spec["weight"])
            elif "shares" in spec:
                if last_prices is None or ticker not in last_prices.index:
                    raise ValueError(
                        f"{ticker}: 'shares' given but no last price available"
                    )
                dollars[ticker] = float(spec["shares"]) * float(last_prices[ticker])
            else:
                raise ValueError(
                    f"{ticker}: holding must specify dollars, shares or weight"
                )
        else:
            dollars[ticker] = float(spec)
    return normalize_weights(dollars, list(holdings.keys()))


# ---------------------------------------------------------------------------
# Portfolio return series
# ---------------------------------------------------------------------------
def portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    kind: Literal["log", "simple"] = "log",
    rebalance: Literal["daily", "none"] = "daily",
) -> pd.Series:
    """Portfolio RETURNS from asset RETURNS and target weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset RETURNS (dates x tickers), log or simple per ``kind``.
    weights : pd.Series
        Target weights, indexed by ticker (must sum to 1).
    kind : {"log", "simple"}
        What ``returns`` are AND what the output is.
    rebalance : {"daily", "none"}
        See module docstring.

    Returns
    -------
    pd.Series
        Portfolio RETURNS named ``"portfolio"``.
    """
    w = weights.reindex(returns.columns)
    if w.isna().any():
        raise KeyError(f"weights missing for: {list(w[w.isna()].index)}")
    simple = log_to_simple(returns) if kind == "log" else returns

    if rebalance == "daily":
        port_simple = simple @ w.values
    elif rebalance == "none":
        growth = (1.0 + simple).cumprod()
        value = growth @ w.values
        port_simple = value / value.shift(1).fillna(1.0) - 1.0
    else:
        raise ValueError("rebalance must be 'daily' or 'none'")

    out = simple_to_log(port_simple) if kind == "log" else port_simple
    out = pd.Series(np.asarray(out, dtype=float), index=returns.index, name="portfolio")
    return out


# ---------------------------------------------------------------------------
# Portfolio object
# ---------------------------------------------------------------------------
@dataclass
class Portfolio:
    """Long-only portfolio on a clean price panel.

    Attributes
    ----------
    prices : pd.DataFrame
        Clean PRICES (output of :func:`risklib.preprocessing.panel.build_panel`).
    returns : pd.DataFrame
        Matching log RETURNS.
    weights : pd.Series
        Target weights, indexed by ticker, summing to 1.
    name : str
    meta : dict
        Proxy backfill details, full-history prices, value, sectors, as-of date.
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.Series
    name: str = "portfolio"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = normalize_weights(self.weights.to_dict(), list(self.prices.columns))

    @classmethod
    def from_prices(
        cls,
        prices: pd.DataFrame,
        holdings: Mapping[str, Any],
        name: str = "portfolio",
        **panel_kwargs,
    ) -> "Portfolio":
        """Build from raw PRICES and a holdings mapping (see holdings_to_weights).

        A holding may carry ``proxy: TICKER``; its history before its first
        real price is then backfilled with the proxy's returns (see
        :func:`risklib.preprocessing.panel.backfill_with_proxy`). The proxy's prices must be
        a column of ``prices``. ``meta["proxied"]`` records what was done.
        """
        tickers = list(holdings.keys())
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise KeyError(f"prices have no column for holdings: {missing}")
        proxies = {t: s["proxy"] for t, s in holdings.items() if isinstance(s, Mapping) and s.get("proxy")}
        missing_p = [p for p in proxies.values() if p not in prices.columns]
        if missing_p:
            raise KeyError(f"prices have no column for proxies: {missing_p}")
        full = prices[tickers].copy()
        proxied: dict[str, dict] = {}
        for t, p in proxies.items():
            full, first = backfill_with_proxy(full, t, prices[p])
            if first is not None:
                proxied[t] = {"proxy": p, "real_from": first, "n_backfilled": int((full.index < first).sum())}
        clean, rets = build_panel(full, **panel_kwargs)
        dropped = [t for t in tickers if t not in clean.columns]
        if dropped:
            raise ValueError(
                f"tickers dropped for short history: {dropped}. "
                "Lower min_obs, remove them, or give them a `proxy:`."
            )
        weights = holdings_to_weights(holdings, last_prices=clean.iloc[-1])
        return cls(
            prices=clean, returns=rets, weights=weights, name=name,
            meta={
                "proxied": proxied,
                "full_prices": full,
                "value": holdings_value(holdings, clean.iloc[-1]),
                "sectors": holdings_sectors(holdings),
                "as_of": clean.index[-1],
            },
        )

    def real_mask(self) -> pd.DataFrame:
        """Boolean frame (dates x tickers): True where the return is real, not proxied."""
        m = pd.DataFrame(True, index=self.returns.index, columns=self.tickers)
        for t, info in self.meta.get("proxied", {}).items():
            if t in m.columns:
                m.loc[m.index < info["real_from"], t] = False
        return m

    def proxy_table(self) -> pd.DataFrame:
        """One row per proxy-backfilled holding (empty if none)."""
        rows = self.meta.get("proxied", {})
        if not rows:
            return pd.DataFrame(columns=["proxy", "real_from", "n_backfilled", "pct_of_sample_proxied"])
        out = pd.DataFrame(rows).T
        out["pct_of_sample_proxied"] = out["n_backfilled"].astype(float) / len(self.meta["full_prices"]) * 100.0
        out.index.name = "ticker"
        return out

    @classmethod
    def from_config(cls, config: Mapping[str, Any], prices: pd.DataFrame) -> "Portfolio":
        """Build from a parsed YAML config dict and raw PRICES."""
        panel_kwargs = dict(config.get("panel", {}))
        return cls.from_prices(
            prices,
            config["holdings"],
            name=config.get("name", "portfolio"),
            **panel_kwargs,
        )

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    def portfolio_returns(
        self, kind: Literal["log", "simple"] = "log", rebalance: Literal["daily", "none"] = "daily"
    ) -> pd.Series:
        """Portfolio RETURNS under the chosen convention (``kind`` = output type)."""
        r = portfolio_returns(self.returns, self.weights, kind="log", rebalance=rebalance)
        return log_to_simple(r) if kind == "simple" else r

    def performance(self, rf_annual: float = 0.0, rebalance: Literal["daily", "none"] = "daily") -> pd.Series:
        """Performance summary of the portfolio's log RETURNS."""
        return performance_summary(self.portfolio_returns("log", rebalance), kind="log", rf_annual=rf_annual)

    def summary(self) -> str:
        w = (self.weights * 100).round(2).astype(str) + "%"
        return f"{self.name}: {self.n_assets} assets, {len(self.returns)} days\n{w.to_string()}"
