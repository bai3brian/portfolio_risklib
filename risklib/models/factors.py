"""Observable factor model built from ETF proxies.

Factors are built from ETF proxies (long or long-minus-short), then each
asset's RETURNS are regressed on the factor RETURNS by OLS with Newey-West
(HAC) standard errors, using statsmodels. The fitted model gives:

* betas, t-stats and R² per asset;
* the split of the portfolio's variance into per-factor systematic parts and
  an idiosyncratic part;
* portfolio factor exposures  βₚ = wᵀ B (used by the stress module).

The default proxy set is in :data:`DEFAULT_FACTORS`. A factor spec is either
a single ticker (long) or a ``[long, short]`` pair (long minus short).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from .. import TRADING_DAYS

# ticker or [long, short]
FactorSpec = str | Sequence[str]

DEFAULT_FACTORS: dict[str, FactorSpec] = {
    "MKT": "SPY",  # US equity market
    "SIZE": ["IWM", "SPY"],  # small minus large
    "VALUE": ["IWD", "IWF"],  # value minus growth (also carries equity "duration")
    "MOM": ["MTUM", "SPY"],  # momentum minus market
    "SEMIS": ["SMH", "SPY"],  # semiconductors minus market
}

# Optional extras. A bond-price factor (TLT) has a near-zero daily beta for
# most equities; rate risk is carried by VALUE instead (see stress.py).
OPTIONAL_FACTORS: dict[str, FactorSpec] = {
    "RATES": "TLT",
    "TECH": ["XLK", "SPY"],
    "ENERGY": ["XLE", "SPY"],
    "GOLD": "GLD",
    "USD": "UUP",
    "CREDIT": ["HYG", "IEF"],
}


def factor_tickers(spec: Mapping[str, FactorSpec] = DEFAULT_FACTORS) -> list[str]:
    """All ETF tickers needed to build the factors in ``spec`` (deduplicated)."""
    out: list[str] = []
    for s in spec.values():
        for t in [s] if isinstance(s, str) else list(s):
            if t not in out:
                out.append(t)
    return out


def build_factor_returns(
    proxy_returns: pd.DataFrame, spec: Mapping[str, FactorSpec] = DEFAULT_FACTORS
) -> pd.DataFrame:
    """Combine ETF proxy RETURNS into factor RETURNS.

    Parameters
    ----------
    proxy_returns : pd.DataFrame
        RETURNS of the ETFs in :func:`factor_tickers` (dates x tickers).
    spec : mapping
        Factor name -> ticker, or -> ``[long, short]``.

    Returns
    -------
    pd.DataFrame
        Factor RETURNS (dates x factors), columns named ``"factor"``.
    """
    cols = {}
    for name, s in spec.items():
        if isinstance(s, str):
            cols[name] = proxy_returns[s]
        else:
            long, short = s
            cols[name] = proxy_returns[long] - proxy_returns[short]
    f = pd.DataFrame(cols)
    f.index.name, f.columns.name = "date", "factor"
    return f


# ---------------------------------------------------------------------------
# OLS with Newey-West standard errors
# ---------------------------------------------------------------------------
def newey_west_lags(n: int) -> int:
    """Default HAC lag length, floor(4 (n/100)^(2/9)) (Newey-West 1994)."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def ols_hac(
    y: np.ndarray, X: np.ndarray, lags: int | None = None, add_const: bool = True
) -> dict:
    """OLS with Newey-West heteroskedasticity/autocorrelation-consistent SEs (statsmodels).

    Parameters
    ----------
    y : ndarray, shape (n,)
    X : ndarray, shape (n, k)
    lags : int or None
        Bartlett-kernel bandwidth. ``None`` uses :func:`newey_west_lags`.
    add_const : bool
        Prepend an intercept column.

    Returns
    -------
    dict with ``coef``, ``se``, ``tstat``, ``pvalue``, ``r2``, ``r2_adj``,
    ``resid``, ``lags``.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    if add_const:
        X = np.column_stack([np.ones(n), X])
    if lags is None:
        lags = newey_west_lags(n)
    # Bartlett-kernel HAC covariance without the small-sample correction; t-based p-values
    res = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}, use_t=True)
    return dict(
        coef=np.asarray(res.params), se=np.asarray(res.bse), tstat=np.asarray(res.tvalues), pvalue=np.asarray(res.pvalues),
        r2=float(res.rsquared), r2_adj=float(res.rsquared_adj), resid=np.asarray(res.resid), lags=lags, cov=np.asarray(res.cov_params()),
    )


# ---------------------------------------------------------------------------
# Factor model
# ---------------------------------------------------------------------------
@dataclass
class FactorModel:
    """Linear factor model  r_i = α_i + Σ_k β_ik f_k + ε_i, fitted by OLS/HAC.

    Call :meth:`fit` with asset RETURNS and factor RETURNS (both daily, same
    kind). All attributes are set by ``fit``.
    """

    lags: int | None = None
    min_real_obs: int = 250

    def fit(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        real_mask: pd.DataFrame | None = None,
    ) -> "FactorModel":
        """Fit one OLS/HAC regression per asset.

        Parameters
        ----------
        asset_returns, factor_returns : pd.DataFrame
            Daily RETURNS on a common calendar.
        real_mask : pd.DataFrame of bool, optional
            Same shape as ``asset_returns``; ``True`` where the return is
            real (not proxy-backfilled). When given, each asset's regression
            uses only its real days, provided there are at least
            ``min_real_obs`` of them; otherwise the full sample is used and
            ``on_real_data`` is False for that asset. The factor covariance
            always uses the full sample.
        """
        idx = asset_returns.index.intersection(factor_returns.index)
        if len(idx) < 60:
            raise ValueError(f"only {len(idx)} overlapping observations; need >= 60")
        R = asset_returns.loc[idx]
        F = factor_returns.loc[idx]
        self.assets = list(R.columns)
        self.factors = list(F.columns)
        self.index = idx
        self.n_obs = len(idx)
        X = F.values
        M = real_mask.reindex(index=idx, columns=self.assets).fillna(True).astype(bool) if real_mask is not None else None

        betas, tstats, alphas, alpha_t, r2, r2_adj, idio = [], [], [], [], [], [], []
        n_real, on_real = [], []
        resids = {}
        for a in self.assets:
            y = R[a].values
            if M is not None and int(M[a].sum()) >= self.min_real_obs and not bool(M[a].all()):
                sel = M[a].values
                fit = ols_hac(y[sel], X[sel], lags=self.lags)
                full_resid = np.full(len(y), np.nan)
                full_resid[sel] = fit["resid"]
                fit["resid"] = full_resid
                on_real.append(True)
                n_real.append(int(sel.sum()))
            else:
                fit = ols_hac(y, X, lags=self.lags)
                on_real.append(M is None or bool(M[a].all()))
                n_real.append(int(M[a].sum()) if M is not None else len(y))
            alphas.append(fit["coef"][0])
            alpha_t.append(fit["tstat"][0])
            betas.append(fit["coef"][1:])
            tstats.append(fit["tstat"][1:])
            r2.append(fit["r2"])
            r2_adj.append(fit["r2_adj"])
            resids[a] = fit["resid"]
            idio.append(float(np.nanstd(fit["resid"], ddof=len(self.factors) + 1)))
            self.hac_lags = fit["lags"]
        self.n_real_obs = pd.Series(n_real, index=self.assets, name="n_real_obs")
        self.on_real_data = pd.Series(on_real, index=self.assets, name="on_real_data")

        self.betas = pd.DataFrame(betas, index=self.assets, columns=self.factors)
        self.tstats = pd.DataFrame(tstats, index=self.assets, columns=self.factors)
        self.alpha = pd.Series(alphas, index=self.assets, name="alpha_daily")
        self.alpha_tstat = pd.Series(alpha_t, index=self.assets, name="alpha_tstat")
        self.r2 = pd.Series(r2, index=self.assets, name="r2")
        self.r2_adj = pd.Series(r2_adj, index=self.assets, name="r2_adj")
        self.idio_vol = pd.Series(idio, index=self.assets, name="idio_vol_daily")
        self.residuals = pd.DataFrame(resids, index=idx)
        self.factor_cov = F.cov(ddof=1)
        self.factor_mean = F.mean()
        for df in (self.betas, self.tstats):
            df.index.name, df.columns.name = "ticker", "factor"
        return self

    def portfolio_exposures(self, weights: pd.Series) -> pd.Series:
        """Portfolio factor betas  βₚ = wᵀ B."""
        w = weights.reindex(self.assets).fillna(0.0).values
        out = pd.Series(w @ self.betas.values, index=self.factors, name="exposure")
        out.index.name = "factor"
        return out

    def risk_decomposition(self, weights: pd.Series, annualize: bool = True) -> pd.DataFrame:
        """Split portfolio variance into per-factor systematic parts + idiosyncratic.

        Uses the *marginal* decomposition  β_k · (F βₚ)_k  so the factor rows
        sum to the total systematic variance (cross terms are allocated).

        Returns
        -------
        pd.DataFrame
            Rows: each factor, ``idiosyncratic``, ``total``. Columns:
            ``variance``, ``vol``, ``pct_of_total``.
        """
        w = weights.reindex(self.assets).fillna(0.0).values
        bp = w @ self.betas.values
        F = self.factor_cov.values
        sys_contrib = bp * (F @ bp)  # sums to bpᵀ F bp
        idio_var = float(w**2 @ self.idio_vol.values**2)
        total = float(sys_contrib.sum() + idio_var)
        scale = TRADING_DAYS if annualize else 1.0
        rows = {f: c for f, c in zip(self.factors, sys_contrib)}
        rows["idiosyncratic"] = idio_var
        rows["total"] = total
        var = pd.Series(rows) * scale
        out = pd.DataFrame(
            {"variance": var, "vol": np.sqrt(var.clip(lower=0)), "pct_of_total": var / (total * scale) * 100.0}
        )
        out.index.name = "source"
        return out

    def portfolio_r2(self, weights: pd.Series) -> float:
        """Share of portfolio variance explained by the factors (0-1)."""
        d = self.risk_decomposition(weights, annualize=False)
        return float(1.0 - d.loc["idiosyncratic", "variance"] / d.loc["total", "variance"])

    def summary(self) -> pd.DataFrame:
        """One row per asset: betas, R², annualized idiosyncratic vol, alpha t-stat."""
        out = self.betas.copy()
        out["r2"] = self.r2
        out["idio_vol_ann"] = self.idio_vol * np.sqrt(TRADING_DAYS)
        out["alpha_ann"] = self.alpha * TRADING_DAYS
        out["alpha_tstat"] = self.alpha_tstat
        out["n_obs"] = self.n_real_obs
        out["real_only"] = self.on_real_data
        return out
