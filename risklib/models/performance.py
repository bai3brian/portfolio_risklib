"""Performance and concentration statistics.

Performance statistics take one return series (``kind`` = "log" or
"simple") and annualize with 252 trading days.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from .. import TRADING_DAYS
from ..preprocessing.panel import log_to_simple, simple_to_log


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------
def drawdown_series(returns: pd.Series, kind: Literal["log", "simple"] = "log") -> pd.Series:
    """Drawdown path (fraction below running peak, <= 0) from RETURNS."""
    r = log_to_simple(returns) if kind == "log" else returns
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(returns: pd.Series, kind: Literal["log", "simple"] = "log") -> float:
    """Maximum drawdown (a negative number) from RETURNS."""
    return float(drawdown_series(returns, kind=kind).min())


def performance_summary(
    returns: pd.Series,
    kind: Literal["log", "simple"] = "log",
    rf_annual: float = 0.0,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """Standard performance table from a RETURNS Series.

    Parameters
    ----------
    returns : pd.Series
        Portfolio RETURNS.
    kind : {"log", "simple"}
        Type of ``returns``.
    rf_annual : float
        Annual risk-free rate used for Sharpe / Sortino (e.g. 0.04).

    Returns
    -------
    pd.Series
        ann_return (CAGR), ann_vol, sharpe, sortino, max_drawdown, calmar,
        hit_rate, skew, kurtosis (excess), n_obs, start, end.
    """
    r = returns.dropna()
    simple = log_to_simple(r) if kind == "log" else r
    logr = r if kind == "log" else simple_to_log(r)
    n = len(r)
    years = n / periods
    total_growth = float(np.exp(logr.sum()))
    cagr = total_growth ** (1.0 / years) - 1.0 if years > 0 else np.nan
    ann_vol = float(simple.std(ddof=1) * np.sqrt(periods))
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    excess = simple - rf_daily
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods)) if excess.std() > 0 else np.nan
    downside = np.minimum(excess, 0.0)
    down_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(periods))
    sortino = float(excess.mean() * periods / down_dev) if down_dev > 0 else np.nan
    mdd = max_drawdown(simple, kind="simple")
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    return pd.Series(
        {
            "ann_return": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": mdd,
            "calmar": calmar,
            "hit_rate": float((simple > 0).mean()),
            "skew": float(simple.skew()),
            "kurtosis": float(simple.kurt()),
            "n_obs": n,
            "start": r.index[0],
            "end": r.index[-1],
        },
        name="performance",
    )


def trailing_performance(
    returns: pd.Series,
    windows: Mapping[str, int] | None = None,
    kind: Literal["log", "simple"] = "log",
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Performance over trailing windows (default 1y, 3y, 5y) and the full sample.

    Windows longer than the sample are skipped.
    """
    windows = dict(windows or {"1y": 252, "3y": 756, "5y": 1260})
    r = returns.dropna()
    rows = {}
    for name, n in windows.items():
        if n <= len(r):
            rows[name] = performance_summary(r.iloc[-n:], kind=kind, rf_annual=rf_annual)
    rows["full"] = performance_summary(r, kind=kind, rf_annual=rf_annual)
    out = pd.DataFrame(rows).T[["ann_return", "ann_vol", "sharpe", "max_drawdown", "n_obs", "start"]]
    out.index.name = "window"
    return out


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------
def concentration(weights: pd.Series, risk_pct: pd.Series | None = None) -> pd.Series:
    """Concentration metrics: Herfindahl index and effective number of positions.

    ``risk_pct`` (each asset's percent of VaR or volatility, summing to 100)
    adds the risk-based versions.
    """
    w = weights / weights.sum()
    hhi_w = float((w**2).sum())
    out = {"hhi_weight": hhi_w, "effective_n_weight": 1.0 / hhi_w, "top1_weight": float(w.max()), "top5_weight": float(w.nlargest(5).sum())}
    if risk_pct is not None:
        r = risk_pct / risk_pct.sum()
        hhi_r = float((r**2).sum())
        out.update({"hhi_risk": hhi_r, "effective_n_risk": 1.0 / hhi_r, "top1_risk": float(r.max()), "top5_risk": float(r.nlargest(5).sum())})
    return pd.Series(out, name="concentration")


def sector_table(weights: pd.Series, sectors: Mapping[str, str], risk_pct: pd.Series | None = None) -> pd.DataFrame:
    """Weight (and risk) share by sector. Unlabelled tickers go to 'Unknown'."""
    sec = pd.Series({t: sectors.get(t, "Unknown") for t in weights.index})
    df = pd.DataFrame({"weight_%": weights * 100.0, "sector": sec})
    if risk_pct is not None:
        df["risk_%"] = risk_pct.reindex(weights.index)
    out = df.groupby("sector").sum(numeric_only=True)
    out["n_holdings"] = df.groupby("sector").size()
    return out.sort_values("weight_%", ascending=False)
