"""Covariance estimators and diagnostics.

All estimators take a RETURNS DataFrame (dates x tickers) and return a
DAILY covariance DataFrame (tickers x tickers). Use :func:`annualize` to
scale by 252.

Estimators
----------
sample_cov          unbiased sample covariance.
ewma_cov            RiskMetrics exponentially weighted covariance (λ=0.94), via pandas.
ledoit_wolf         Ledoit-Wolf (2004) shrinkage toward a scaled identity, via
                    sklearn.covariance.LedoitWolf.

Helpers
-------
annualize, is_psd, nearest_psd, portfolio_vol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf

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
    """RiskMetrics EWMA covariance:  Σ_t = λ Σ_{t-1} + (1-λ) r_t r_tᵀ, seeded with the sample covariance.

    Returns the estimate as of the day *after* the last observation. Returns
    are demeaned by their sample mean; the recursion runs through pandas'
    exponentially weighted mean on the cross-products.
    """
    r = returns.dropna()
    e = r.values - r.values.mean(axis=0)
    k = e.shape[1]
    cross = np.einsum("ti,tj->tij", e, e).reshape(len(e), k * k)
    seeded = np.vstack([np.cov(e, rowvar=False, ddof=1).reshape(1, k * k), cross])
    last = pd.DataFrame(seeded).ewm(alpha=1.0 - lam, adjust=False).mean().iloc[-1].values
    return _frame(last.reshape(k, k), r.columns)


def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2004) shrinkage toward μI with the analytic optimal δ (scikit-learn).

    Σ_shrunk = (1-δ) S + δ μ I,  μ = tr(S)/p, rescaled from the maximum-likelihood
    to the unbiased (n-1) convention used by :func:`sample_cov`.

    Returns
    -------
    (cov, shrinkage)
    """
    r = returns.dropna()
    n = len(r)
    lw = LedoitWolf(assume_centered=False).fit(r.values)
    return _frame(lw.covariance_ * n / (n - 1), r.columns), float(lw.shrinkage_)


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
