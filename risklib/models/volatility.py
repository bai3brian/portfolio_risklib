"""Univariate volatility models and forecast evaluation.

All models implement the :class:`VolatilityModel` protocol:

    model.fit(returns)            -> self          (returns: a RETURNS Series)
    model.conditional_variance()  -> pd.Series     (in-sample σ²_t)
    model.forecast(horizon)       -> np.ndarray    (σ²_{T+1..T+h})

Models
------
EWMA        RiskMetrics exponentially weighted variance, fixed λ (0.94 daily),
            computed with ``pandas.Series.ewm``.
GARCH       GARCH(1,1) by maximum likelihood with the ``arch`` package
            (normal or standardized Student-t innovations).
GJRGARCH    GJR-GARCH(1,1): adds a leverage term for negative shocks.

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
from arch import arch_model
from scipy import stats

from .. import TRADING_DAYS

# arch is fitted on percent returns, the scale its optimizer is tuned for;
# parameters and variances are converted back to return units.
_SCALE = 100.0


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
    """RiskMetrics EWMA:  σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}, seeded with the sample variance.

    The forecast is flat: σ²_{T+h} = σ²_{T+1} for all h (no mean reversion).
    """

    lam: float = 0.94
    demean: bool = True

    def fit(self, returns: pd.Series) -> "EWMA":
        r = returns.dropna().astype(float)
        self.index = r.index
        self.mu = float(r.mean()) if self.demean else 0.0
        e = (r - self.mu).values
        # pandas' recursive EWM (adjust=False) on [σ²_0, e²_0, ..., e²_{T-2}] gives σ²_0..σ²_{T-1}
        x = pd.Series(np.concatenate([[np.var(e, ddof=1)], e[:-1] ** 2]))
        self.sigma2 = x.ewm(alpha=1.0 - self.lam, adjust=False).mean().values
        self._last_e2 = e[-1] ** 2
        self.params = pd.Series({"lambda": self.lam, "mu": self.mu})
        return self

    def conditional_variance(self) -> pd.Series:
        return pd.Series(self.sigma2, index=self.index, name="sigma2")

    def forecast(self, horizon: int = 1) -> np.ndarray:
        nxt = self.lam * self.sigma2[-1] + (1.0 - self.lam) * self._last_e2
        return np.full(horizon, nxt)


# ---------------------------------------------------------------------------
# GARCH family (arch package)
# ---------------------------------------------------------------------------
@dataclass
class GARCH:
    """GARCH(1,1):  σ²_t = ω + α ε²_{t-1} + β σ²_{t-1},  fitted by maximum likelihood with ``arch``.

    Parameters
    ----------
    dist : {"normal", "t"}
        Innovation distribution. ``"t"`` is arch's standardized Student-t and
        estimates the degrees of freedom ν.
    demean : bool
        Subtract the sample mean before fitting (μ is not jointly estimated;
        for daily equity data the difference is negligible).

    After ``fit``: ``params`` (Series of ω, α, β, [γ], [ν] in return units),
    ``se``, ``tstats``, ``loglik``, ``aic``, ``bic``, ``persistence``
    (α + β + γ/2), ``half_life`` (days), ``unconditional_vol_ann``, and
    ``theta`` (the parameters as an array in that order).
    """

    dist: Literal["normal", "t"] = "normal"
    demean: bool = True
    asym: bool = field(default=False, init=False, repr=False)

    @property
    def _names(self) -> list[str]:
        names = ["omega", "alpha", "beta"] + (["gamma"] if self.asym else [])
        return names + (["nu"] if self.dist == "t" else [])

    def _arch(self, e: pd.Series):
        return arch_model(e * _SCALE, mean="Zero", vol="GARCH", p=1, o=1 if self.asym else 0, q=1,
                          dist="t" if self.dist == "t" else "normal", rescale=False)

    def fit(self, returns: pd.Series) -> "GARCH":
        r = returns.dropna().astype(float)
        self.mu = float(r.mean()) if self.demean else 0.0
        e = r - self.mu
        res = self._arch(e).fit(disp="off", show_warning=False)
        self._arch_params = res.params
        arch_names = {"omega": "omega", "alpha": "alpha[1]", "beta": "beta[1]", "gamma": "gamma[1]", "nu": "nu"}
        unit = {"omega": _SCALE**2}
        self.params = pd.Series({k: float(res.params[arch_names[k]]) / unit.get(k, 1.0) for k in self._names})
        self.se = pd.Series({k: float(res.std_err[arch_names[k]]) / unit.get(k, 1.0) for k in self._names})
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tstats = self.params / self.se
        self.theta = self.params.values.copy()
        n, k = len(e), len(self.params)
        self.loglik = float(res.loglikelihood) + n * np.log(_SCALE)  # density of returns, not percent returns
        self.aic = 2 * k - 2 * self.loglik
        self.bic = k * np.log(n) - 2 * self.loglik
        g = self.params.get("gamma", 0.0)
        self.persistence = float(self.params["alpha"] + self.params["beta"] + 0.5 * g)
        self.half_life = float(np.log(0.5) / np.log(self.persistence)) if 0 < self.persistence < 1 else np.inf
        uncond = self.params["omega"] / (1.0 - self.persistence) if self.persistence < 1 else np.nan
        self.unconditional_vol_ann = float(np.sqrt(uncond * TRADING_DAYS)) if uncond == uncond else np.nan
        self._set_path(e, res)
        return self

    def _set_path(self, e: pd.Series, res) -> None:
        self.index = e.index
        self.e = e.values
        self.sigma2 = (np.asarray(res.conditional_volatility, float) / _SCALE) ** 2
        self._res = res

    def update(self, returns: pd.Series) -> "GARCH":
        """Re-run the variance filter on new data with the fitted parameters held fixed."""
        e = returns.dropna().astype(float) - self.mu
        self._set_path(e, self._arch(e).fix(self._arch_params))
        return self

    def conditional_variance(self) -> pd.Series:
        return pd.Series(self.sigma2, index=self.index, name="sigma2")

    def standardized_residuals(self) -> pd.Series:
        return pd.Series(self.e / np.sqrt(self.sigma2), index=self.index, name="z")

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """σ²_{T+1..T+h} (arch's analytic forecast; GJR uses E[1(ε<0)] = ½ beyond one step)."""
        f = self._res.forecast(horizon=horizon, reindex=False)
        return np.asarray(f.variance.values[-1], float) / _SCALE**2

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
    return model.update(r)


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
