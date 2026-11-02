"""Out-of-sample VaR and ES backtesting.

Workflow
--------
1. :func:`rolling_var` produces genuinely out-of-sample daily VaR (and ES)
   forecasts: at each day t the model sees data up to t-1 only.
2. A breach is a day when the realised loss exceeds the VaR forecast.
3. Tests on the breach series:

   kupiec_pof              unconditional coverage: is the breach rate = 1-α?
   christoffersen_ind      independence: do breaches cluster?
   conditional_coverage    both at once (LR_uc + LR_ind).
   basel_traffic_light     the regulatory zone for 250-day windows.
   acerbi_szekely          ES test (Z₂): are losses on breach days as large
                           as the ES said they would be?

4. :func:`backtest_table` runs all of it for several models and returns
   one comparison table.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from . import risk
from .volatility import EWMA, GARCH, GJRGARCH

Method = Literal["normal", "t", "historical", "hist_weighted", "ewma", "garch_normal", "gjr_normal", "garch_t", "fhs"]

# accepted aliases
_ALIASES = {"garch": "garch_normal", "gjr": "gjr_normal"}

# DEFAULT_METHODS is what the report backtests; ALL_METHODS lists every option for
# ``backtest.methods`` in the config (garch_t is by far the slowest).
ALL_METHODS: tuple[str, ...] = ("normal", "t", "historical", "hist_weighted", "ewma", "garch_normal", "gjr_normal", "garch_t", "fhs")
DEFAULT_METHODS: tuple[str, ...] = ("normal", "historical", "garch_normal", "fhs")


# ---------------------------------------------------------------------------
# rolling forecasts
# ---------------------------------------------------------------------------
def rolling_var(
    returns: pd.Series,
    method: Method = "historical",
    alpha: float = 0.99,
    window: int = 500,
    refit_every: int = 21,
    n_sims: int = 5_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-sample 1-day VaR and ES forecasts.

    Parameters
    ----------
    returns : pd.Series
        Portfolio SIMPLE RETURNS.
    method : str
        "normal"         rolling-window normal (σ = sample std)
        "t"              rolling-window Student-t
        "historical"     empirical quantile of the window
        "hist_weighted"  age-weighted historical (decay 0.99)
        "ewma"           normal quantile with EWMA σ (λ=0.94), updated daily
        "garch_normal"   normal quantile with a GARCH(1,1) σ forecast
        "gjr_normal"     normal quantile with a GJR-GARCH σ forecast
        "garch_t"        Student-t quantile with a GARCH(1,1) σ forecast
                         (ν estimated jointly) — isolates the tail assumption
        "fhs"            filtered historical simulation (GARCH + bootstrap of
                         standardized residuals)
        ("garch" and "gjr" are accepted as aliases of the *_normal rows.)
    window : int
        Estimation window (days).

    Returns
    -------
    pd.DataFrame
        Columns ``VaR``, ``ES`` (positive losses), ``loss`` (realised −r),
        ``breach`` (bool). Index: days ``window`` onward.
    """
    method = _ALIASES.get(method, method)
    r = returns.dropna().astype(float)
    n = len(r)
    idx = r.index[window:]
    var = np.empty(n - window)
    es = np.empty(n - window)
    model = None
    rng = np.random.default_rng(seed)

    for i, t in enumerate(range(window, n)):
        hist = r.iloc[t - window : t]
        need_refit = model is None or (i % refit_every == 0)
        if method == "normal":
            var[i], es[i] = risk.parametric_normal(hist, alpha)
        elif method == "t":
            if need_refit:
                model = stats.t.fit(hist.values, floc=0.0)
            nu, loc, scale = model
            q = stats.t.ppf(alpha, nu)
            var[i] = q * scale
            es[i] = scale * stats.t.pdf(q, nu) * (nu + q**2) / ((nu - 1) * (1 - alpha))
        elif method == "historical":
            var[i], es[i] = risk.historical(hist, alpha)
        elif method == "hist_weighted":
            var[i], es[i] = risk.historical(hist, alpha, decay=0.99)
        elif method == "ewma":
            m = EWMA(0.94).fit(hist)
            var[i], es[i] = risk.parametric_normal(hist, alpha, sigma=np.sqrt(m.forecast(1)[0]))
        elif method in ("garch_normal", "gjr_normal"):
            if need_refit:
                model = (GJRGARCH() if method == "gjr_normal" else GARCH()).fit(hist)
            else:
                from .volatility import _roll_forward

                model = _roll_forward(model, hist)
            var[i], es[i] = risk.parametric_normal(hist, alpha, sigma=np.sqrt(model.forecast(1)[0]))
        elif method == "garch_t":
            if need_refit:
                model = GARCH(dist="t").fit(hist)
            else:
                from .volatility import _roll_forward

                model = _roll_forward(model, hist)
            sig = np.sqrt(model.forecast(1)[0])
            nu = float(model.params["nu"])
            # standardized-t (unit variance) quantile and ES
            q = stats.t.ppf(alpha, nu)
            k = np.sqrt((nu - 2.0) / nu)
            var[i] = q * k * sig
            es[i] = k * sig * stats.t.pdf(q, nu) * (nu + q**2) / ((nu - 1) * (1 - alpha))
        elif method == "fhs":
            if need_refit:
                model = GARCH().fit(hist)
            else:
                from .volatility import _roll_forward

                model = _roll_forward(model, hist)
            z = model.standardized_residuals().values
            z = z - z.mean()
            sim = model.mu + np.sqrt(model.forecast(1)[0]) * rng.choice(z, size=n_sims)
            var[i], es[i] = risk.var_es_from_scenarios(sim, alpha)
        else:
            raise ValueError(f"unknown method {method!r}")

    loss = -r.iloc[window:].values
    out = pd.DataFrame({"VaR": var, "ES": es, "loss": loss, "breach": loss > var}, index=idx)
    out.index.name = "date"
    out.attrs.update(method=method, alpha=alpha, window=window)
    return out


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def kupiec_pof(breaches: np.ndarray | pd.Series, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: P(breach) = 1-α.  LR ~ χ²(1).
    """
    b = np.asarray(breaches, bool)
    n, x = len(b), int(b.sum())
    p = 1.0 - alpha
    phat = x / n if n else np.nan
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * np.log(p)
    else:
        lr = -2.0 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return {"n": n, "breaches": x, "expected": n * p, "breach_rate": phat, "LR_uc": float(lr), "p_value": float(stats.chi2.sf(lr, 1))}


def christoffersen_ind(breaches: np.ndarray | pd.Series) -> dict:
    """Christoffersen (1998) independence test: breaches should not cluster.

    Models breaches as a 2-state Markov chain; H0: π₀₁ = π₁₁.  LR ~ χ²(1).
    """
    b = np.asarray(breaches, int)
    prev, cur = b[:-1], b[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _ll(p, ones, zeros):
        if p in (0.0, 1.0):
            return 0.0 if (ones == 0 or zeros == 0) else -np.inf
        return ones * np.log(p) + zeros * np.log(1 - p)

    ll0 = _ll(pi, n01 + n11, n00 + n10)
    ll1 = _ll(pi01, n01, n00) + _ll(pi11, n11, n10)
    lr = -2.0 * (ll0 - ll1)
    lr = max(float(lr), 0.0) if np.isfinite(lr) else np.nan
    return {"n01": n01, "n11": n11, "pi01": pi01, "pi11": pi11, "LR_ind": lr, "p_value": float(stats.chi2.sf(lr, 1)) if lr == lr else np.nan}


def conditional_coverage(breaches: np.ndarray | pd.Series, alpha: float) -> dict:
    """Christoffersen conditional coverage: LR_cc = LR_uc + LR_ind ~ χ²(2)."""
    k = kupiec_pof(breaches, alpha)
    c = christoffersen_ind(breaches)
    lr = k["LR_uc"] + (c["LR_ind"] if c["LR_ind"] == c["LR_ind"] else 0.0)
    return {"LR_cc": float(lr), "p_value": float(stats.chi2.sf(lr, 2))}


def basel_traffic_light(breaches: np.ndarray | pd.Series, alpha: float = 0.99, window: int = 250) -> dict:
    """Basel traffic-light zone from the LAST ``window`` days (99% VaR).

    Green: ≤4 breaches, Yellow: 5-9, Red: ≥10 (for 250 days at 99%).
    For other α/window, zones are set at the binomial 95% / 99.99% points.
    """
    b = np.asarray(breaches, bool)[-window:]
    x = int(b.sum())
    n = len(b)
    p = 1 - alpha
    if abs(alpha - 0.99) < 1e-9 and n == 250:
        green, red = 4, 10
    else:
        green = int(stats.binom.ppf(0.95, n, p))
        red = int(stats.binom.ppf(0.9999, n, p)) + 1
    zone = "green" if x <= green else ("yellow" if x < red else "red")
    return {"breaches": x, "n": n, "zone": zone, "green_max": green, "red_min": red, "cum_prob": float(stats.binom.cdf(x, n, p))}


def acerbi_szekely(loss: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float, n_boot: int = 2000, seed: int = 0) -> dict:
    """Acerbi-Székely (2014) ES test Z₂ = Σ L_t 1{L_t>VaR_t} / (T (1-α) ES_t) − 1.

    H0: Z₂ = 0 (ES correct). With losses positive, Z₂ > 0 ⇒ realized losses
    beyond VaR exceeded what ES predicted (ES too small); Z₂ < 0 ⇒ ES was
    conservative. One-sided p-value (ES too small) from a bootstrap under
    H0 that rescales breach-day losses to have mean ES.
    """
    loss, var, es = (np.asarray(a, float) for a in (loss, var, es))
    T = len(loss)
    hit = loss > var
    z2 = float(np.sum(loss[hit] / es[hit]) / (T * (1 - alpha)) - 1.0)
    exceed_ratio = float((loss[hit] / es[hit]).mean()) if hit.any() else np.nan
    # bootstrap null: breaches occur w.p. 1-α, exceedance/ES has mean 1
    rng = np.random.default_rng(seed)
    ratios = loss[hit] / es[hit] if hit.any() else np.array([1.0])
    ratios0 = ratios / ratios.mean()  # impose mean 1 under H0
    boot = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.binomial(T, 1 - alpha)
        boot[i] = rng.choice(ratios0, size=k, replace=True).sum() / (T * (1 - alpha)) - 1.0 if k else -1.0
    p = float((boot >= z2).mean())
    return {"Z2": z2, "p_value": p, "mean_exceed_over_es": exceed_ratio, "breaches": int(hit.sum())}


# ---------------------------------------------------------------------------
# comparison table
# ---------------------------------------------------------------------------
def backtest_table(
    returns: pd.Series,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    alpha: float = 0.99,
    window: int = 500,
    refit_every: int = 21,
    n_sims: int = 5_000,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest several VaR models and tabulate the tests.

    Returns
    -------
    (table, forecasts)
        ``table``: one row per method with breach counts, breach rate,
        Kupiec/Christoffersen/CC p-values, traffic-light zone, ES Z₂ and
        average VaR. ``forecasts``: the per-method rolling DataFrames.
    """
    rows, fc = [], {}
    for m in methods:
        m = _ALIASES.get(m, m)
        f = rolling_var(returns, m, alpha=alpha, window=window, refit_every=refit_every, n_sims=n_sims)
        fc[m] = f
        k = kupiec_pof(f["breach"], alpha)
        c = christoffersen_ind(f["breach"])
        cc = conditional_coverage(f["breach"], alpha)
        tl = basel_traffic_light(f["breach"], alpha)
        tl_full = basel_traffic_light(f["breach"], alpha, window=len(f))
        es = acerbi_szekely(f["loss"], f["VaR"], f["ES"], alpha)
        rows.append(
            {
                "method": m,
                "n_days": k["n"],
                "breaches": k["breaches"],
                "expected": k["expected"],
                "breach_rate_%": k["breach_rate"] * 100,
                "kupiec_p": k["p_value"],
                "christoffersen_p": c["p_value"],
                "cc_p": cc["p_value"],
                "traffic_light_250d": tl["zone"],
                "traffic_light_full": tl_full["zone"],
                "es_Z2": es["Z2"],
                "es_p": es["p_value"],
                "avg_VaR_%": f["VaR"].mean() * 100,
                "max_loss_%": f["loss"].max() * 100,
            }
        )
    table = pd.DataFrame(rows).set_index("method")
    table.attrs.update(alpha=alpha, window=window)
    return table, fc
