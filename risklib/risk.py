"""Value-at-Risk, Expected Shortfall, Monte Carlo simulation and risk attribution.

Conventions: inputs are portfolio simple returns unless noted, losses are
positive, and ``alpha`` is the confidence level (0.95, 0.975, 0.99). VaR_α is
the loss exceeded with probability 1 − α; ES_α is the mean loss beyond it.

VaR / ES methods
    parametric_normal     zero-mean normal, √h scaling
    parametric_t          Student-t fitted by MLE, closed-form ES
    historical            empirical quantile, optional age weighting (BRW 1998)
    filtered_historical   GARCH-filtered historical simulation (FHS)
    monte_carlo           any engine from :func:`simulate`

Monte Carlo engines (``dist``)
    normal            multivariate normal
    t                 multivariate Student-t, covariance-matched
    gaussian_copula   Gaussian dependence with empirical marginals
    t_copula          Student-t dependence (tail dependence) with empirical marginals

Attribution
    component_var_parametric   Euler decomposition of normal VaR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .volatility import GARCH

DEFAULT_ALPHAS = (0.99, 0.975)

# Rows of the VaR/ES table. DEFAULT_METHODS is what the report shows; ALL_METHODS
# lists every option for ``risk.methods`` in the config.
ALL_METHODS = ("normal", "t", "historical", "hist_weighted", "fhs", "mc_normal", "mc_t", "mc_t_copula")
DEFAULT_METHODS = ("normal", "historical", "fhs", "mc_t_copula")

Dist = Literal["normal", "t", "gaussian_copula", "t_copula"]


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------
def var_es_from_scenarios(pnl: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> tuple[float, float]:
    """VaR and ES from a vector of portfolio returns, optionally probability-weighted."""
    loss = -np.asarray(pnl, float)
    if weights is None:
        var = float(np.quantile(loss, alpha, method="linear"))
        tail = loss[loss >= var]
        es = float(tail.mean()) if tail.size else var
        return var, es
    w = np.asarray(weights, float)
    w = w / w.sum()
    order = np.argsort(loss)
    l_sorted, w_sorted = loss[order], w[order]
    cum = np.cumsum(w_sorted)
    i = int(np.searchsorted(cum, alpha))
    var = float(l_sorted[min(i, len(loss) - 1)])
    tail_w = np.where(l_sorted >= var, w_sorted, 0.0)
    es = float((tail_w * l_sorted).sum() / tail_w.sum()) if tail_w.sum() > 0 else var
    return var, es


def overlapping_sum(x: pd.Series, h: int) -> pd.Series:
    """Overlapping h-day simple returns from daily simple returns."""
    if h == 1:
        return x
    return (1.0 + x).rolling(h).apply(np.prod, raw=True).dropna() - 1.0


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------
def parametric_normal(returns: pd.Series, alpha: float = 0.99, horizon: int = 1, sigma: float | None = None, zero_mean: bool = True) -> tuple[float, float]:
    """Normal VaR/ES. ``sigma`` (daily) may be supplied, e.g. a GARCH forecast."""
    r = returns.dropna()
    s = float(r.std(ddof=1)) if sigma is None else float(sigma)
    m = 0.0 if zero_mean else float(r.mean())
    z = stats.norm.ppf(alpha)
    sh = s * np.sqrt(horizon)
    mh = m * horizon
    var = z * sh - mh
    es = sh * stats.norm.pdf(z) / (1.0 - alpha) - mh
    return float(var), float(es)


def parametric_t(returns: pd.Series, alpha: float = 0.99, horizon: int = 1, zero_mean: bool = True) -> tuple[float, float, float]:
    """Student-t VaR/ES with ν and scale fitted by MLE. Returns (VaR, ES, ν)."""
    r = returns.dropna().values
    nu, loc, scale = stats.t.fit(r, floc=0.0) if zero_mean else stats.t.fit(r)
    nu = max(nu, 2.05)
    q = stats.t.ppf(alpha, nu)
    sh = scale * np.sqrt(horizon)
    mh = loc * horizon
    var = q * sh - mh
    es = sh * stats.t.pdf(q, nu) * (nu + q**2) / ((nu - 1.0) * (1.0 - alpha)) - mh
    return float(var), float(es), float(nu)


# ---------------------------------------------------------------------------
# Historical
# ---------------------------------------------------------------------------
def historical(returns: pd.Series, alpha: float = 0.99, horizon: int = 1, window: int | None = None, decay: float | None = None) -> tuple[float, float]:
    """Historical-simulation VaR/ES.

    ``window`` keeps only the last ``window`` days; ``decay`` (e.g. 0.98)
    age-weights scenarios by decay^age (Boudoukh, Richardson & Whitelaw 1998).
    """
    r = returns.dropna()
    if window is not None:
        r = r.iloc[-window:]
    rh = overlapping_sum(r, horizon)
    w = None
    if decay is not None:
        ages = np.arange(len(rh))[::-1]
        w = decay**ages
    return var_es_from_scenarios(rh.values, alpha, w)


def filtered_historical(
    returns: pd.Series,
    alpha: float = 0.99,
    horizon: int = 1,
    model: GARCH | None = None,
    n_sims: int = 10_000,
    seed: int | None = 0,
) -> tuple[float, float, GARCH]:
    """Filtered historical simulation (Barone-Adesi, Giannopoulos & Vosper 1999).

    Fits a GARCH model, bootstraps its standardized residuals and runs them
    through the GARCH recursion from σ²_{T+1}. Returns (VaR, ES, fitted model).
    """
    r = returns.dropna()
    if model is not None and hasattr(model, "theta") and len(model.e) == len(r):
        m = model
    else:
        m = (model or GARCH()).fit(r)
    z = m.standardized_residuals().values
    z = z - z.mean()
    rng = np.random.default_rng(seed)
    omega, a, b = m.theta[:3]
    g = m.theta[3] if m.asym else 0.0
    s2 = np.full(n_sims, m.forecast(1)[0])
    cum = np.zeros(n_sims)
    for _ in range(horizon):
        zz = rng.choice(z, size=n_sims, replace=True)
        e = np.sqrt(s2) * zz
        cum += m.mu + e
        s2 = omega + (a + g * (e < 0)) * e**2 + b * s2
    var, es = var_es_from_scenarios(cum, alpha)
    return var, es, m


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def _chol(cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, 1e-12)
        return np.linalg.cholesky(V @ np.diag(w) @ V.T)


def _mvt(rng, L: np.ndarray, nu: float, size: tuple) -> np.ndarray:
    """Covariance-matched multivariate t: z · sqrt((ν−2)/ν / W), W ~ χ²_ν/ν."""
    k = L.shape[0]
    z = rng.standard_normal(size + (k,)) @ L.T
    w = rng.chisquare(nu, size=size) / nu
    return z * np.sqrt((nu - 2.0) / nu / w)[..., None]


def _empirical_ppf(sample: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse empirical CDF of ``sample`` at probabilities ``u``."""
    return np.quantile(sample, np.clip(u, 1e-6, 1 - 1e-6), method="linear")


def simulate(
    returns: pd.DataFrame,
    n_sims: int = 10_000,
    horizon: int = 1,
    dist: Dist = "normal",
    nu: float = 5.0,
    cov: pd.DataFrame | None = None,
    mean: pd.Series | None = None,
    zero_mean: bool = True,
    seed: int | None = 0,
) -> np.ndarray:
    """Simulate daily log returns, shape (n_sims, horizon, k).

    ``returns`` (historical log returns) supplies the empirical marginals for
    the copula engines and the mean/covariance when ``mean``/``cov`` are not
    given. ``nu`` is the degrees of freedom for the "t" and "t_copula" engines.
    """
    rng = np.random.default_rng(seed)
    R = returns.dropna()
    k = R.shape[1]
    S = (cov.reindex(index=R.columns, columns=R.columns).values if cov is not None else np.cov(R.values, rowvar=False, ddof=1))
    mu = np.zeros(k) if zero_mean else (mean.reindex(R.columns).values if mean is not None else R.values.mean(axis=0))
    L = _chol(S)
    size = (n_sims, horizon)

    if dist == "normal":
        x = rng.standard_normal(size + (k,)) @ L.T
    elif dist == "t":
        if nu <= 2:
            raise ValueError("nu must be > 2 for finite variance")
        x = _mvt(rng, L, nu, size)
    elif dist in ("gaussian_copula", "t_copula"):
        sd = np.sqrt(np.diag(S))
        corr = S / np.outer(sd, sd)
        Lc = _chol(corr)
        if dist == "gaussian_copula":
            z = rng.standard_normal(size + (k,)) @ Lc.T
            u = stats.norm.cdf(z)
        else:
            z = _mvt(rng, Lc, nu, size) * np.sqrt(nu / (nu - 2.0))
            u = stats.t.cdf(z, df=nu)
        demeaned = R.values - R.values.mean(axis=0)
        x = np.empty(size + (k,))
        for j in range(k):
            x[..., j] = _empirical_ppf(demeaned[:, j], u[..., j])
        if cov is not None:  # match the marginal variances to the supplied covariance
            x *= (sd / demeaned.std(axis=0, ddof=1))[None, None, :]
    else:
        raise ValueError(f"unknown dist {dist!r}")
    return x + mu


def horizon_returns(sim: np.ndarray) -> np.ndarray:
    """Sum daily log returns over the horizon: (n_sims, horizon, k) → (n_sims, k)."""
    return sim.sum(axis=1)


def portfolio_pnl(sim_h: np.ndarray, weights: pd.Series, columns) -> np.ndarray:
    """Portfolio simple return per simulation from h-day log returns (n_sims, k)."""
    w = weights.reindex(columns).fillna(0.0).values
    return (np.exp(sim_h) - 1.0) @ w


def simulate_portfolio(returns: pd.DataFrame, weights: pd.Series, n_sims: int = 10_000, horizon: int = 1, dist: Dist = "normal", **kwargs) -> np.ndarray:
    """Simulated h-day portfolio simple returns, shape (n_sims,)."""
    sim = simulate(returns, n_sims=n_sims, horizon=horizon, dist=dist, **kwargs)
    return portfolio_pnl(horizon_returns(sim), weights, returns.columns)


def simulate_paths(returns: pd.DataFrame, weights: pd.Series, n_sims: int = 1000, horizon: int = 21, dist: Dist = "normal", **kwargs) -> np.ndarray:
    """Daily-rebalanced portfolio wealth paths, shape (n_sims, horizon + 1), starting at 1."""
    sim = simulate(returns, n_sims=n_sims, horizon=horizon, dist=dist, **kwargs)
    w = weights.reindex(returns.columns).fillna(0.0).values
    daily = (np.exp(sim) - 1.0) @ w
    wealth = np.cumprod(1.0 + daily, axis=1)
    return np.concatenate([np.ones((n_sims, 1)), wealth], axis=1)


def monte_carlo(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.99,
    horizon: int = 1,
    dist: Dist = "normal",
    n_sims: int = 20_000,
    cov: pd.DataFrame | None = None,
    nu: float = 5.0,
    seed: int | None = 0,
) -> tuple[float, float, np.ndarray]:
    """Monte Carlo VaR/ES from asset log returns. Returns (VaR, ES, simulated P&L)."""
    pnl = simulate_portfolio(asset_returns, weights, n_sims=n_sims, horizon=horizon, dist=dist, cov=cov, nu=nu, seed=seed)
    var, es = var_es_from_scenarios(pnl, alpha)
    return var, es, pnl


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
def component_var_parametric(cov: pd.DataFrame, weights: pd.Series, alpha: float = 0.99, horizon: int = 1) -> pd.DataFrame:
    """Euler decomposition of normal VaR: CVaR_i = w_i ∂VaR/∂w_i = z w_i (Σw)_i / σ_p.

    Columns: weight, marginal_var, component_var, pct_of_var, standalone_var.
    Component VaRs sum to the portfolio VaR.
    """
    w = weights.reindex(cov.columns).fillna(0.0).values
    S = cov.values * horizon
    sig_p = float(np.sqrt(w @ S @ w))
    z = stats.norm.ppf(alpha)
    marginal = z * (S @ w) / sig_p
    comp = w * marginal
    standalone = z * np.sqrt(np.diag(S)) * w
    out = pd.DataFrame(
        {"weight": w, "marginal_var": marginal, "component_var": comp, "pct_of_var": comp / comp.sum() * 100.0, "standalone_var": standalone},
        index=cov.columns,
    )
    out.index.name = "ticker"
    out.attrs["portfolio_var"] = float(comp.sum())
    out.attrs["diversification_benefit"] = float(standalone.sum() - comp.sum())
    return out


# ---------------------------------------------------------------------------
# Estimation uncertainty
# ---------------------------------------------------------------------------
def bootstrap_ci_historical(returns: pd.Series, alpha: float = 0.99, n_boot: int = 300, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap confidence interval for 1-day historical VaR and ES (i.i.d. resampling of days)."""
    r = returns.dropna().values
    rng = np.random.default_rng(seed)
    n = len(r)
    vs, es_ = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        samp = r[rng.integers(0, n, n)]
        vs[b], es_[b] = var_es_from_scenarios(samp, alpha)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    var, es = var_es_from_scenarios(r, alpha)
    return {"VaR": var, "VaR_lo": float(np.quantile(vs, lo)), "VaR_hi": float(np.quantile(vs, hi)),
            "ES": es, "ES_lo": float(np.quantile(es_, lo)), "ES_hi": float(np.quantile(es_, hi)), "n_boot": n_boot}


def bootstrap_ci_fhs(
    returns: pd.Series, alpha: float = 0.99, model: GARCH | None = None, n_boot: int = 200,
    n_sims: int = 5_000, ci: float = 0.95, seed: int = 0,
) -> dict:
    """Bootstrap confidence interval for 1-day FHS VaR and ES.

    Resamples the standardized residuals before each simulation; GARCH
    parameters are held fixed.
    """
    r = returns.dropna()
    m = model if (model is not None and hasattr(model, "theta") and len(model.e) == len(r)) else GARCH().fit(r)
    z = m.standardized_residuals().values
    z = z - z.mean()
    sig = np.sqrt(m.forecast(1)[0])
    rng = np.random.default_rng(seed)
    vs, es_ = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        zb = z[rng.integers(0, len(z), len(z))]
        sim = m.mu + sig * rng.choice(zb, size=n_sims, replace=True)
        vs[b], es_[b] = var_es_from_scenarios(sim, alpha)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    var, es, _ = filtered_historical(r, alpha, 1, model=m, n_sims=max(n_sims, 20_000), seed=seed)
    return {"VaR": var, "VaR_lo": float(np.quantile(vs, lo)), "VaR_hi": float(np.quantile(vs, hi)),
            "ES": es, "ES_lo": float(np.quantile(es_, lo)), "ES_hi": float(np.quantile(es_, hi)), "n_boot": n_boot}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
@dataclass
class RiskTable:
    """VaR/ES for the chosen methods at every horizon and confidence level, in one table."""

    alphas: tuple = DEFAULT_ALPHAS
    horizons: tuple = (1, 10)
    methods: tuple = DEFAULT_METHODS
    n_sims: int = 20_000
    seed: int = 0
    n_boot: int = 300

    def run(self, port_simple: pd.Series, asset_log: pd.DataFrame, weights: pd.Series, cov: pd.DataFrame | None = None, portfolio_value: float | None = None) -> pd.DataFrame:
        """Build the table, indexed by (method, horizon, alpha).

        Multi-day historical rows are √h-scaled from the 1-day estimate, since
        an overlapping 10-day 99% quantile rests on only a few independent
        windows. Age-weighted HS is reported at 1 day only. The 1-day
        historical and FHS rows carry bootstrap 95% confidence intervals.
        """
        unknown = [m for m in self.methods if m not in ALL_METHODS]
        if unknown:
            raise ValueError(f"unknown VaR methods {unknown}; choose from {ALL_METHODS}")
        nan4 = (np.nan,) * 4
        rows = []
        garch_fit = None
        for h in self.horizons:
            for a in self.alphas:
                for meth in self.methods:
                    if meth == "normal":
                        v, e = parametric_normal(port_simple, a, h)
                        rows.append(("Parametric normal", h, a, v, e, *nan4))
                    elif meth == "t":
                        v, e, nu = parametric_t(port_simple, a, h)
                        rows.append((f"Parametric t (ν={nu:.1f})", h, a, v, e, *nan4))
                    elif meth == "historical":
                        if h == 1:
                            b = bootstrap_ci_historical(port_simple, a, n_boot=self.n_boot, seed=self.seed)
                            rows.append(("Historical", h, a, b["VaR"], b["ES"], b["VaR_lo"], b["VaR_hi"], b["ES_lo"], b["ES_hi"]))
                        else:
                            v1, e1 = historical(port_simple, a, 1)
                            rows.append((f"Historical (√{h}-scaled from 1-day)", h, a, v1 * np.sqrt(h), e1 * np.sqrt(h), *nan4))
                    elif meth == "hist_weighted":
                        if h == 1:
                            v, e = historical(port_simple, a, 1, decay=0.99)
                            rows.append(("Historical age-weighted (0.99)", h, a, v, e, *nan4))
                    elif meth == "fhs":
                        v, e, garch_fit = filtered_historical(port_simple, a, h, model=garch_fit, n_sims=self.n_sims, seed=self.seed)
                        if h == 1:
                            b = bootstrap_ci_fhs(port_simple, a, model=garch_fit, n_boot=max(self.n_boot // 2, 50), seed=self.seed)
                            rows.append(("Filtered historical (GARCH)", h, a, v, e, b["VaR_lo"], b["VaR_hi"], b["ES_lo"], b["ES_hi"]))
                        else:
                            rows.append(("Filtered historical (GARCH)", h, a, v, e, *nan4))
                    else:
                        dist, label = {"mc_normal": ("normal", "MC normal"), "mc_t": ("t", "MC Student-t (ν=5)"), "mc_t_copula": ("t_copula", "MC t-copula + empirical")}[meth]
                        v, e, _ = monte_carlo(asset_log, weights, a, h, dist=dist, n_sims=self.n_sims, cov=cov, seed=self.seed)
                        rows.append((label, h, a, v, e, *nan4))
        out = pd.DataFrame(rows, columns=["method", "horizon", "alpha", "VaR", "ES", "VaR_lo", "VaR_hi", "ES_lo", "ES_hi"])
        if portfolio_value:
            out["VaR_$"] = out["VaR"] * portfolio_value
            out["ES_$"] = out["ES"] * portfolio_value
        return out.set_index(["method", "horizon", "alpha"])
