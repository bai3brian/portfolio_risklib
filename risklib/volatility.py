"""Univariate volatility models and forecast evaluation.

All models implement the :class:`VolatilityModel` protocol:

    model.fit(returns)            -> self          (returns: a RETURNS Series)
    model.conditional_variance()  -> pd.Series     (in-sample σ²_t)
    model.forecast(horizon)       -> np.ndarray    (σ²_{T+1..T+h})

Models
------
EWMA        RiskMetrics exponentially weighted variance, fixed λ (0.94 daily).
GARCH       GARCH(1,1) by maximum likelihood (normal or Student-t innovations).
GJRGARCH    GJR-GARCH(1,1): adds a leverage term for negative shocks.

The recursions and likelihoods are implemented directly in numpy/scipy; the
``arch`` package is optional and only used by :func:`crosscheck_arch`.

Evaluation
----------
:func:`qlike`, :func:`mse` score variance forecasts against a realized proxy
(squared demeaned returns by default). :func:`diebold_mariano` tests whether
two forecast-loss series differ. :func:`rolling_forecast` produces true
out-of-sample one-step forecasts by refitting on a moving window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import optimize, signal, special, stats

from .data import TRADING_DAYS


@runtime_checkable
class VolatilityModel(Protocol):
    def fit(self, returns: pd.Series) -> "VolatilityModel": ...
    def conditional_variance(self) -> pd.Series: ...
    def forecast(self, horizon: int = 1) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------
@dataclass
class EWMA:
    """RiskMetrics EWMA:  σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}.

    The forecast is flat: σ²_{T+h} = σ²_{T+1} for all h (no mean reversion).
    """

    lam: float = 0.94
    demean: bool = True

    def fit(self, returns: pd.Series) -> "EWMA":
        r = returns.dropna().astype(float)
        self.index = r.index
        self.mu = float(r.mean()) if self.demean else 0.0
        e = (r - self.mu).values
        n = len(e)
        s2 = np.empty(n)
        s2[0] = float(np.var(e, ddof=1))
        for t in range(1, n):
            s2[t] = self.lam * s2[t - 1] + (1.0 - self.lam) * e[t - 1] ** 2
        self.sigma2 = s2
        self._last_e2 = e[-1] ** 2
        self.params = pd.Series({"lambda": self.lam, "mu": self.mu})
        return self

    def conditional_variance(self) -> pd.Series:
        return pd.Series(self.sigma2, index=self.index, name="sigma2")

    def forecast(self, horizon: int = 1) -> np.ndarray:
        nxt = self.lam * self.sigma2[-1] + (1.0 - self.lam) * self._last_e2
        return np.full(horizon, nxt)


# ---------------------------------------------------------------------------
# GARCH family (MLE)
# ---------------------------------------------------------------------------
def _garch_recursion(e: np.ndarray, omega: float, alpha: float, beta: float, gamma: float, s2_0: float) -> np.ndarray:
    """σ²_t = ω + (α + γ·1[ε_{t-1}<0]) ε²_{t-1} + β σ²_{t-1},  σ²_0 = s2_0.

    The recursion is a first-order linear filter in σ², so it is evaluated
    with ``scipy.signal.lfilter`` rather than a Python loop.
    """
    n = len(e)
    s2 = np.empty(n)
    s2[0] = s2_0
    if n > 1:
        e_lag = e[:-1]
        x = omega + (alpha + gamma * (e_lag < 0.0)) * e_lag**2
        s2[1:], _ = signal.lfilter([1.0], [1.0, -beta], x, zi=[beta * s2_0])
    return s2


def _neg_loglik(theta: np.ndarray, e: np.ndarray, s2_0: float, asym: bool, dist: str) -> float:
    if asym:
        omega, alpha, beta, gamma = theta[:4]
        rest = theta[4:]
    else:
        omega, alpha, beta = theta[:3]
        gamma = 0.0
        rest = theta[3:]
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + gamma < 0 or alpha + beta + 0.5 * gamma >= 0.9999:
        return 1e10
    s2 = _garch_recursion(e, omega, alpha, beta, gamma, s2_0)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    if dist == "normal":
        ll = -0.5 * (np.log(2.0 * np.pi) + np.log(s2) + e**2 / s2)
    else:  # student-t, unit-variance scaling
        nu = rest[0]
        if nu <= 2.05 or nu > 200:
            return 1e10
        c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        ll = c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(e**2 / (s2 * (nu - 2)))
    return float(-ll.sum())


def _numerical_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    k = len(x)
    H = np.zeros((k, k))
    h = eps * np.maximum(np.abs(x), 1e-3)
    for i in range(k):
        for j in range(i, k):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
    return H


@dataclass
class GARCH:
    """GARCH(1,1):  σ²_t = ω + α ε²_{t-1} + β σ²_{t-1},  fitted by MLE.

    Parameters
    ----------
    dist : {"normal", "t"}
        Innovation distribution. ``"t"`` estimates the degrees of freedom ν.
    demean : bool
        Subtract the sample mean before fitting (μ is not jointly estimated;
        for daily equity data the difference is negligible).

    After ``fit``: ``params`` (Series), ``se``, ``tstats``, ``loglik``,
    ``aic``, ``bic``, ``persistence`` (α+β), ``half_life`` (days),
    ``unconditional_vol_ann``.
    """

    dist: Literal["normal", "t"] = "normal"
    demean: bool = True
    asym: bool = field(default=False, init=False, repr=False)

    @property
    def _names(self) -> list[str]:
        names = ["omega", "alpha", "beta"] + (["gamma"] if self.asym else [])
        return names + (["nu"] if self.dist == "t" else [])

    def fit(self, returns: pd.Series) -> "GARCH":
        r = returns.dropna().astype(float)
        self.index = r.index
        self.mu = float(r.mean()) if self.demean else 0.0
        e = (r - self.mu).values
        self.e = e
        var = float(np.var(e, ddof=1))
        self.s2_0 = var

        # Optimize over scaled parameters u = (ω/σ², α, β, [γ], [ν]) with the
        # log-likelihood divided by n, so all parameters are of similar size.
        n = len(e)
        k_extra = (1 if self.asym else 0) + (1 if self.dist == "t" else 0)
        scale = np.array([var, 1.0, 1.0] + [1.0] * k_extra)
        full = lambda u: np.asarray(u, float) * scale  # noqa: E731

        def obj_u(u):
            return _neg_loglik(full(u), e, self.s2_0, self.asym, self.dist) / n

        def stationarity(u):
            g = u[3] if self.asym else 0.0
            return 0.9999 - (u[1] + u[2] + 0.5 * g)

        bounds = [(1e-8, 1.0), (0.0, 1.0), (0.0, 1.0)] + ([(0.0, 1.0)] if self.asym else []) + (
            [(2.1, 200.0)] if self.dist == "t" else [])

        # start grid over (α, β, γ, ν); ω/σ² from variance targeting
        ab_grid = [(0.05, 0.90), (0.10, 0.85), (0.02, 0.97), (0.15, 0.75), (0.08, 0.91)]
        g_grid = [0.0, 0.05, 0.15] if self.asym else [None]
        nu_grid = [4.0, 6.0, 8.0, 12.0] if self.dist == "t" else [None]
        starts = []
        for a, b in ab_grid:
            for g in g_grid:
                for nu in nu_grid:
                    gg = 0.0 if g is None else g
                    if a + b + 0.5 * gg >= 0.999:
                        continue
                    u = [1.0 - a - b - 0.5 * gg, a, b] + ([gg] if self.asym else []) + (
                        [nu] if self.dist == "t" else [])
                    starts.append(u)

        best = None
        cons = [{"type": "ineq", "fun": stationarity}]
        for u0 in starts:
            try:
                res = optimize.minimize(obj_u, np.array(u0, float), method="SLSQP", bounds=bounds, constraints=cons,
                                        options={"maxiter": 500, "ftol": 1e-12})
            except Exception:  # pragma: no cover
                continue
            if res.fun < 1e8 and (best is None or res.fun < best.fun):
                best = res
        # polish with a derivative-free search
        if best is not None:
            pol = optimize.minimize(obj_u, best.x, method="Nelder-Mead",
                                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
            if pol.fun < best.fun and stationarity(pol.x) > 0 and np.all(pol.x >= np.array([b[0] for b in bounds])):
                best = pol
        u_hat = best.x
        theta = full(u_hat)
        self.theta = theta
        self.loglik = -float(best.fun) * n
        self.sigma2 = _garch_recursion(e, theta[0], theta[1], theta[2], theta[3] if self.asym else 0.0, self.s2_0)
        self.params = pd.Series(theta, index=self._names)

        # standard errors: inverse Hessian in scaled space, mapped back by the delta method
        try:
            H_u = _numerical_hessian(lambda u: obj_u(u) * n, u_hat)
            cov_u = np.linalg.pinv(H_u)
            cov = cov_u * np.outer(scale, scale)
            se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        except Exception:  # pragma: no cover
            se = np.full(len(theta), np.nan)
        self.se = pd.Series(se, index=self._names)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tstats = self.params / self.se
        k = len(theta)
        self.aic = 2 * k - 2 * self.loglik
        self.bic = k * np.log(n) - 2 * self.loglik
        g = theta[3] if self.asym else 0.0
        self.persistence = float(theta[1] + theta[2] + 0.5 * g)
        self.half_life = float(np.log(0.5) / np.log(self.persistence)) if 0 < self.persistence < 1 else np.inf
        uncond = theta[0] / (1.0 - self.persistence) if self.persistence < 1 else np.nan
        self.unconditional_vol_ann = float(np.sqrt(uncond * TRADING_DAYS)) if uncond == uncond else np.nan
        return self

    def conditional_variance(self) -> pd.Series:
        return pd.Series(self.sigma2, index=self.index, name="sigma2")

    def standardized_residuals(self) -> pd.Series:
        return pd.Series(self.e / np.sqrt(self.sigma2), index=self.index, name="z")

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """σ²_{T+1..T+h}. For h>1 uses E[ε²]=σ² and E[1(ε<0)]=½ (symmetric innovations)."""
        omega, alpha, beta = self.theta[:3]
        g = self.theta[3] if self.asym else 0.0
        out = np.empty(horizon)
        e_last, s2_last = self.e[-1], self.sigma2[-1]
        lev = g if e_last < 0 else 0.0
        out[0] = omega + (alpha + lev) * e_last**2 + beta * s2_last
        for h in range(1, horizon):
            out[h] = omega + (alpha + 0.5 * g + beta) * out[h - 1]
        return out

    def forecast_cumulative_variance(self, horizon: int) -> float:
        """Σ_{h=1..H} σ²_{T+h}: the variance of the H-day return under the model."""
        return float(self.forecast(horizon).sum())

    def summary(self) -> pd.DataFrame:
        out = pd.DataFrame({"estimate": self.params, "std_err": self.se, "t_stat": self.tstats})
        out.index.name = "parameter"
        return out


@dataclass
class GJRGARCH(GARCH):
    """GJR-GARCH(1,1):  σ²_t = ω + (α + γ·1[ε_{t-1}<0]) ε²_{t-1} + β σ²_{t-1}.

    γ > 0 means negative shocks raise next-day variance more than positive
    ones (the leverage effect). Persistence is α + β + γ/2.
    """

    def __post_init__(self) -> None:
        self.asym = True


# ---------------------------------------------------------------------------
# Forecast evaluation
# ---------------------------------------------------------------------------
def realized_proxy(returns: pd.Series, demean: bool = True) -> pd.Series:
    """Realized-variance proxy for daily data: squared (demeaned) return."""
    r = returns.dropna().astype(float)
    e = r - r.mean() if demean else r
    return (e**2).rename("rv")


def qlike(forecast: np.ndarray | pd.Series, realized: np.ndarray | pd.Series) -> float:
    """QLIKE loss  mean( RV/σ̂² − ln(RV/σ̂²) − 1 ). Robust to noisy RV proxies."""
    f = np.asarray(forecast, float)
    rv = np.asarray(realized, float)
    ratio = rv / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse(forecast: np.ndarray | pd.Series, realized: np.ndarray | pd.Series) -> float:
    """Mean squared error of a variance forecast against the RV proxy."""
    f = np.asarray(forecast, float)
    rv = np.asarray(realized, float)
    return float(np.mean((f - rv) ** 2))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int | None = None) -> dict:
    """Diebold-Mariano test on the loss differential d = loss_a − loss_b.

    Negative statistic ⇒ model A has lower loss. Uses a HAC variance with
    Bartlett weights.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if lags is None:
        lags = int(np.floor(n ** (1 / 3)))
    dbar = d.mean()
    dc = d - dbar
    var = float(dc @ dc) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(dc[lag:] @ dc[:-lag]) / n
    dm = dbar / np.sqrt(var / n) if var > 0 else np.nan
    p = 2.0 * stats.norm.sf(abs(dm)) if dm == dm else np.nan
    return {"dm_stat": float(dm), "p_value": float(p), "mean_loss_diff": float(dbar), "n": n}


def rolling_forecast(
    model_factory,
    returns: pd.Series,
    window: int = 1000,
    refit_every: int = 21,
    start: int | None = None,
) -> pd.Series:
    """True out-of-sample one-step-ahead variance forecasts.

    The model is refit every ``refit_every`` days on the trailing ``window``
    observations; between refits the fitted parameters are kept and the
    recursion is simply rolled forward with the new data.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning a fresh model, e.g. ``lambda: GARCH()``.
    returns : pd.Series
        RETURNS.
    window : int
        Estimation window length.
    refit_every : int
        Days between refits.

    Returns
    -------
    pd.Series
        Forecast σ²_t for each day t >= window, made with data up to t-1.
    """
    r = returns.dropna().astype(float)
    n = len(r)
    start = window if start is None else max(start, window)
    out = pd.Series(np.nan, index=r.index[start:], name="sigma2_forecast")
    model = None
    for t in range(start, n):
        if model is None or (t - start) % refit_every == 0:
            model = model_factory().fit(r.iloc[t - window : t])
        else:
            model = _roll_forward(model, r.iloc[t - window : t])
        out.iloc[t - start] = model.forecast(1)[0]
    return out


def _roll_forward(model, r: pd.Series):
    """Keep fitted parameters, recompute the variance path on new data."""
    if isinstance(model, EWMA):
        return EWMA(lam=model.lam, demean=model.demean).fit(r)
    e = (r - model.mu).values
    model.e = e
    model.index = r.index
    g = model.theta[3] if model.asym else 0.0
    model.sigma2 = _garch_recursion(e, model.theta[0], model.theta[1], model.theta[2], g, model.s2_0)
    return model


def compare_models(
    returns: pd.Series,
    models: dict | None = None,
    window: int = 1000,
    refit_every: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample comparison of volatility models on one RETURNS series.

    Returns
    -------
    (scores, forecasts)
        ``scores``: one row per model with QLIKE, MSE and the DM test vs the
        first model. ``forecasts``: the forecast paths (dates x models).
    """
    if models is None:
        models = {"EWMA(0.94)": lambda: EWMA(0.94), "GARCH(1,1)": lambda: GARCH(), "GJR-GARCH": lambda: GJRGARCH()}
    rv = realized_proxy(returns)
    fc = {}
    for name, factory in models.items():
        fc[name] = rolling_forecast(factory, returns, window=window, refit_every=refit_every)
    forecasts = pd.DataFrame(fc)
    rv = rv.reindex(forecasts.index)
    base = list(models)[0]
    rows = []
    base_loss = None
    for name in forecasts.columns:
        f = forecasts[name].values
        ql = rv.values / f - np.log(rv.values / f) - 1.0
        if base_loss is None:
            base_loss = ql
        dm = diebold_mariano(ql, base_loss) if name != base else {"dm_stat": np.nan, "p_value": np.nan}
        rows.append({"model": name, "qlike": float(ql.mean()), "mse": mse(f, rv), "dm_vs_" + base: dm["dm_stat"], "dm_pvalue": dm["p_value"]})
    scores = pd.DataFrame(rows).set_index("model")
    return scores, forecasts


def crosscheck_arch(returns: pd.Series) -> pd.DataFrame | None:
    """Compare the GARCH(1,1) fit with the ``arch`` package, if it is installed."""
    try:
        from arch import arch_model
    except ImportError:
        return None
    r = returns.dropna() * 100.0  # arch expects percent returns
    am = arch_model(r, vol="GARCH", p=1, q=1, dist="normal", mean="Constant").fit(disp="off")
    ours = GARCH().fit(returns)
    theirs = pd.Series({"omega": am.params["omega"] / 1e4, "alpha": am.params["alpha[1]"], "beta": am.params["beta[1]"]})
    return pd.DataFrame({"risklib": ours.params[["omega", "alpha", "beta"]], "arch": theirs})
