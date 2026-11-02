"""Covariance estimators and diagnostics.

All estimators take a RETURNS DataFrame (dates x tickers) and return a
DAILY covariance DataFrame (tickers x tickers). Use :func:`annualize` to
scale by 252.

Estimators
----------
sample_cov          unbiased sample covariance.
ewma_cov            RiskMetrics exponentially weighted covariance (λ=0.94).
ledoit_wolf         Ledoit-Wolf (2004) shrinkage toward a scaled identity;
                    matches sklearn.covariance.LedoitWolf.

Helpers
-------
annualize, is_psd, nearest_psd, portfolio_vol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def _frame(S: np.ndarray, cols) -> pd.DataFrame:
    out = pd.DataFrame(S, index=cols, columns=cols)
    out.index.name = out.columns.name = "ticker"
    return out


def annualize(cov: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
    """Scale a daily covariance to annual (× periods)."""
    return cov * periods


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Unbiased sample covariance of RETURNS."""
    r = returns.dropna()
    return _frame(np.cov(r.values, rowvar=False, ddof=1), r.columns)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics EWMA covariance:  Σ_t = λ Σ_{t-1} + (1-λ) r_{t-1} r_{t-1}ᵀ.

    Returns the estimate as of the day *after* the last observation.
    """
    r = returns.dropna()
    e = r.values - r.values.mean(axis=0)
    S = np.cov(e, rowvar=False, ddof=1)
    for t in range(len(e)):
        S = lam * S + (1.0 - lam) * np.outer(e[t], e[t])
    return _frame(S, r.columns)


def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2004) shrinkage toward μI, with analytic shrinkage δ*.

    Σ_shrunk = (1-δ) S + δ μ I,  μ = tr(S)/p.

    Returns
    -------
    (cov, shrinkage)
    """
    r = returns.dropna()
    X = r.values - r.values.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    delta = np.sum((S - mu * np.eye(p)) ** 2)
    # β̄ = (1/n²) Σ_t ||x_t x_tᵀ − S||²_F
    beta_bar = 0.0
    for t in range(n):
        xx = np.outer(X[t], X[t])
        beta_bar += np.sum((xx - S) ** 2)
    beta_bar /= n**2
    shrink = float(np.clip(beta_bar / delta, 0.0, 1.0)) if delta > 0 else 1.0
    S_shrunk = (1.0 - shrink) * S + shrink * mu * np.eye(p)
    # rescale to the unbiased convention used by sample_cov
    S_shrunk *= n / (n - 1)
    return _frame(S_shrunk, r.columns), shrink


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def is_psd(cov: pd.DataFrame, tol: float = 1e-10) -> bool:
    """True if the smallest eigenvalue is at least ``-tol``."""
    return bool(np.linalg.eigvalsh(cov.values).min() >= -tol)


def nearest_psd(cov: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    """Clip negative eigenvalues to ``eps`` (simple PSD repair)."""
    w, V = np.linalg.eigh((cov.values + cov.values.T) / 2)
    w = np.maximum(w, eps)
    return _frame(V @ np.diag(w) @ V.T, cov.columns)


def portfolio_vol(cov: pd.DataFrame, weights: pd.Series, annualize_: bool = True) -> float:
    """√(wᵀ Σ w), annualized by default."""
    w = weights.reindex(cov.columns).fillna(0.0).values
    v = float(w @ cov.values @ w)
    return float(np.sqrt(v * (TRADING_DAYS if annualize_ else 1.0)))
