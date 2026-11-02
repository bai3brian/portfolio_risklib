"""Long-only, fully invested portfolio optimization and walk-forward evaluation.

Objectives
----------
min_variance      argmin wᵀΣw
max_sharpe        argmax (wᵀμ − r_f) / √(wᵀΣw)
target_vol        max wᵀμ  s.t. √(wᵀΣw) ≤ σ*
risk_parity       equal risk contribution (Spinu 2013 convex form)

All take a DAILY covariance (any estimator from :mod:`risklib.covariance`)
and return a weight Series. Constraints: Σw = 1, ``w_min ≤ w_i ≤ w_max``.

:func:`walk_forward` re-estimates and re-optimizes on a rolling window and
reports only out-of-sample results, comparing covariance estimators and
strategies with equal weight and the current weights.

:func:`expected_returns` offers ``"mean"``, ``"shrunk"`` (James-Stein toward
the grand mean) or ``"zero"``. Only max-Sharpe and target-vol use them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize

from . import covariance as cv
from .data import TRADING_DAYS
from .portfolio import performance_summary

ExpRet = Literal["mean", "shrunk", "zero"]


def expected_returns(returns: pd.DataFrame, method: ExpRet = "shrunk") -> pd.Series:
    """Daily expected returns estimated from log returns."""
    mu = returns.mean()
    if method == "mean":
        return mu
    if method == "zero":
        return mu * 0.0
    # James-Stein toward the cross-sectional grand mean
    k = len(mu)
    n = len(returns)
    grand = mu.mean()
    var_mu = returns.var(ddof=1) / n  # variance of each mean estimate
    diff2 = ((mu - grand) ** 2).sum()
    shrink = float(np.clip((k - 3) * var_mu.mean() / diff2, 0.0, 1.0)) if k > 3 and diff2 > 0 else 1.0
    return (1 - shrink) * mu + shrink * grand


def _solve(objective: Callable, k: int, w_min: float, w_max: float, extra_cons: list | None = None, x0: np.ndarray | None = None) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + (extra_cons or [])
    bounds = [(w_min, w_max)] * k
    x0 = np.full(k, 1.0 / k) if x0 is None else x0
    res = optimize.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, w_min, w_max)
    return w / w.sum()


def _series(w: np.ndarray, cols) -> pd.Series:
    s = pd.Series(w, index=cols, name="weight")
    s.index.name = "ticker"
    return s


def min_variance(cov: pd.DataFrame, w_min: float = 0.0, w_max: float = 1.0) -> pd.Series:
    """Minimum-variance weights, argmin wᵀΣw."""
    S = cov.values
    return _series(_solve(lambda w: w @ S @ w, len(S), w_min, w_max), cov.columns)


def max_sharpe(cov: pd.DataFrame, mu: pd.Series, rf_daily: float = 0.0, w_min: float = 0.0, w_max: float = 1.0) -> pd.Series:
    """Maximum-Sharpe (tangency) weights."""
    S = cov.values
    m = mu.reindex(cov.columns).values - rf_daily

    def neg_sharpe(w):
        v = np.sqrt(w @ S @ w)
        return -(w @ m) / v if v > 0 else 0.0

    return _series(_solve(neg_sharpe, len(S), w_min, w_max), cov.columns)


def target_vol(cov: pd.DataFrame, mu: pd.Series, vol_annual: float, w_min: float = 0.0, w_max: float = 1.0) -> pd.Series:
    """Max expected return subject to annualized vol ≤ ``vol_annual``.

    If even the min-variance portfolio exceeds the target, returns min-variance.
    """
    S = cov.values
    m = mu.reindex(cov.columns).values
    v_target_daily2 = (vol_annual**2) / TRADING_DAYS
    mv = min_variance(cov, w_min, w_max)
    if mv.values @ S @ mv.values >= v_target_daily2:
        return mv
    cons = [{"type": "ineq", "fun": lambda w: v_target_daily2 - w @ S @ w}]
    return _series(_solve(lambda w: -(w @ m), len(S), w_min, w_max, cons, x0=mv.values), cov.columns)


def risk_parity(cov: pd.DataFrame, budgets: pd.Series | None = None, w_min: float = 0.0, w_max: float = 1.0) -> pd.Series:
    """Risk budgeting (equal risk contribution by default).

    Solves the convex problem  min ½ wᵀΣw − Σ b_i ln(w_i)  then normalizes;
    the solution satisfies  w_i (Σw)_i ∝ b_i  (Spinu 2013). Bounds are then
    imposed by a second SLSQP pass on the risk-contribution mismatch.
    """
    S = cov.values
    k = len(S)
    b = np.full(k, 1.0 / k) if budgets is None else budgets.reindex(cov.columns).values / budgets.sum()

    def obj(w):
        return 0.5 * w @ S @ w - b @ np.log(np.maximum(w, 1e-12))

    res = optimize.minimize(obj, np.full(k, 1.0 / k), method="L-BFGS-B", bounds=[(1e-8, None)] * k)
    w = res.x / res.x.sum()
    if w_min > 0 or w_max < 1:
        def mismatch(w):
            rc = w * (S @ w)
            rc = rc / rc.sum()
            return ((rc - b) ** 2).sum() * 1e4

        w = _solve(mismatch, k, w_min, w_max, x0=np.clip(w, w_min, w_max))
    return _series(w, cov.columns)


def risk_contributions(cov: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Per-asset volatility contribution  w_i (Σw)_i / σ_p  (sums to σ_p, annualized)."""
    w = weights.reindex(cov.columns).fillna(0.0).values
    S = cov.values
    sig = np.sqrt(w @ S @ w)
    rc = w * (S @ w) / sig * np.sqrt(TRADING_DAYS)
    out = pd.DataFrame({"weight": w, "vol_contribution": rc, "pct_of_risk": rc / rc.sum() * 100.0}, index=cov.columns)
    out.index.name = "ticker"
    return out


def efficient_frontier(cov: pd.DataFrame, mu: pd.Series, n_points: int = 25, w_min: float = 0.0, w_max: float = 1.0) -> pd.DataFrame:
    """Annualized (vol, return) pairs along the long-only frontier."""
    S = cov.values
    m = mu.reindex(cov.columns).values
    mv = min_variance(cov, w_min, w_max)
    r_lo = float(mv.values @ m)
    r_hi = float(m.max()) if w_max >= 1.0 else float(np.sort(m)[::-1][: int(np.ceil(1 / w_max))].mean())
    rows = []
    for r_t in np.linspace(r_lo, r_hi, n_points):
        cons = [{"type": "eq", "fun": lambda w, r_t=r_t: w @ m - r_t}]
        w = _solve(lambda w: w @ S @ w, len(S), w_min, w_max, cons, x0=mv.values)
        rows.append({"ann_vol": np.sqrt(w @ S @ w * TRADING_DAYS), "ann_return": (w @ m) * TRADING_DAYS})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-forward out-of-sample evaluation
# ---------------------------------------------------------------------------
def walk_forward(
    returns: pd.DataFrame,
    current_weights: pd.Series | None = None,
    window: int = 500,
    rebalance_every: int = 63,
    estimators: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    strategies: tuple[str, ...] = ("min_variance", "risk_parity"),
    w_min: float = 0.0,
    w_max: float = 1.0,
    exp_ret: ExpRet = "shrunk",
    rf_annual: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Roll through history: estimate on ``window`` days, hold ``rebalance_every``.

    Returns
    -------
    (summary, oos_returns, weights_history)
        ``summary``: performance table per (estimator, strategy) plus
        ``equal_weight`` and ``current`` benchmarks, with average turnover.
        ``oos_returns``: daily out-of-sample log returns per column.
        ``weights_history``: weights at each rebalance (MultiIndex).
    """
    if estimators is None:
        estimators = {"sample": cv.sample_cov, "ledoit_wolf": lambda r: cv.ledoit_wolf(r)[0]}
    R = returns.dropna()
    n, k = R.shape
    cols = list(R.columns)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    simple = np.exp(R.values) - 1

    names = [f"{e}/{s}" for e in estimators for s in strategies] + ["equal_weight"] + (["current"] if current_weights is not None else [])
    oos = {nm: [] for nm in names}
    turnover = {nm: [] for nm in names}
    prev_w = {nm: None for nm in names}
    w_hist = []
    dates = []

    for start in range(window, n, rebalance_every):
        est = R.iloc[start - window : start]
        end = min(start + rebalance_every, n)
        block = simple[start:end]
        mu = expected_returns(est, exp_ret)
        for nm in names:
            if nm == "equal_weight":
                w = np.full(k, 1.0 / k)
            elif nm == "current":
                w = current_weights.reindex(cols).fillna(0.0).values
            else:
                e, s = nm.split("/")
                S = estimators[e](est)
                if s == "min_variance":
                    w = min_variance(S, w_min, w_max).values
                elif s == "risk_parity":
                    w = risk_parity(S, None, w_min, w_max).values
                elif s == "max_sharpe":
                    w = max_sharpe(S, mu, rf_daily, w_min, w_max).values
                else:
                    raise ValueError(f"unknown strategy {s!r}")
            oos[nm].append(np.log1p(block @ w))
            if prev_w[nm] is not None:
                turnover[nm].append(float(np.abs(w - prev_w[nm]).sum()))
            prev_w[nm] = w
            w_hist.append(pd.Series(w, index=cols, name=(R.index[start], nm)))
        dates.append(R.index[start])

    idx = R.index[window : window + sum(len(v) for v in oos["equal_weight"])]
    oos_df = pd.DataFrame({nm: np.concatenate(v) for nm, v in oos.items()}, index=idx)
    rows = {}
    for nm in names:
        perf = performance_summary(oos_df[nm], kind="log", rf_annual=rf_annual)
        rows[nm] = perf[["ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar"]]
        rows[nm]["avg_turnover"] = float(np.mean(turnover[nm])) if turnover[nm] else 0.0
    summary = pd.DataFrame(rows).T
    summary.index.name = "portfolio"
    weights_history = pd.DataFrame(w_hist)
    weights_history.index = pd.MultiIndex.from_tuples(weights_history.index, names=["rebalance_date", "portfolio"])
    return summary, oos_df, weights_history
