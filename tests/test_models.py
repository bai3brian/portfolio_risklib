"""Tests for factors, volatility and covariance.

Synthetic data with known true parameters, so estimators are checked for
parameter recovery rather than only for running.
"""

import numpy as np
import pandas as pd
import pytest

from risklib import covariance as cv
from risklib import factors as fm
from risklib import synthetic
from risklib import volatility as vol


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------
def test_factor_tickers_dedup_and_order():
    spec = {"MKT": "SPY", "SIZE": ["IWM", "SPY"], "RATES": "TLT"}
    assert fm.factor_tickers(spec) == ["SPY", "IWM", "TLT"]


def test_build_factor_returns_long_short():
    idx = pd.bdate_range("2020-01-01", periods=3)
    proxies = pd.DataFrame({"SPY": [0.01, 0.02, -0.01], "IWM": [0.02, 0.0, 0.0]}, index=idx)
    f = fm.build_factor_returns(proxies, {"MKT": "SPY", "SIZE": ["IWM", "SPY"]})
    assert f["SIZE"].iloc[0] == pytest.approx(0.01)
    assert f["SIZE"].iloc[2] == pytest.approx(0.01)


def test_ols_hac_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.standard_normal((n, 2))
    y = 0.5 + 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.standard_normal(n) * 0.5
    fit = fm.ols_hac(y, X)
    np.testing.assert_allclose(fit["coef"], [0.5, 1.2, -0.7], atol=0.05)
    assert fit["r2"] > 0.8
    assert abs(fit["tstat"][1]) > 20


def test_factor_model_recovers_betas_and_decomposition_sums():
    R, F, true_b = synthetic.factor_panel(n=2500, n_assets=5, n_factors=3, seed=3)
    model = fm.FactorModel().fit(R, F)
    np.testing.assert_allclose(model.betas.values, true_b.values, atol=0.08)
    assert (model.r2 > 0.3).all()
    w = pd.Series(1 / 5, index=R.columns)
    dec = model.risk_decomposition(w, annualize=False)
    factor_rows = dec.loc[model.factors, "variance"].sum()
    assert factor_rows + dec.loc["idiosyncratic", "variance"] == pytest.approx(dec.loc["total", "variance"])
    assert 0 < model.portfolio_r2(w) < 1


def test_portfolio_exposures_are_weighted_betas():
    R, F, _ = synthetic.factor_panel(n=600, n_assets=3, n_factors=2, seed=4)
    model = fm.FactorModel().fit(R, F)
    w = pd.Series([0.5, 0.3, 0.2], index=R.columns)
    exp = model.portfolio_exposures(w)
    np.testing.assert_allclose(exp.values, w.values @ model.betas.values)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def test_ewma_recursion_matches_hand_computation():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, -0.01], index=pd.bdate_range("2020-01-01", periods=5))
    m = vol.EWMA(lam=0.9, demean=False).fit(r)
    s2 = m.conditional_variance().values
    assert s2[1] == pytest.approx(0.9 * s2[0] + 0.1 * 0.01**2)
    assert m.forecast(3).shape == (3,)
    assert m.forecast(1)[0] == pytest.approx(0.9 * s2[-1] + 0.1 * 0.01**2)


def test_garch_mle_recovers_true_parameters():
    r, s2_true = synthetic.garch_series(4000, omega=2e-6, alpha=0.08, beta=0.90, seed=7)
    m = vol.GARCH().fit(pd.Series(r))
    assert m.params["alpha"] == pytest.approx(0.08, abs=0.03)
    assert m.params["beta"] == pytest.approx(0.90, abs=0.04)
    assert m.persistence < 1.0
    assert np.corrcoef(m.conditional_variance().values, s2_true)[0, 1] > 0.9
    assert m.se.notna().all()
    assert isinstance(m, vol.VolatilityModel)


def test_gjr_finds_leverage_when_present():
    r, _ = synthetic.garch_series(4000, omega=2e-6, alpha=0.03, beta=0.90, gamma=0.10, seed=8)
    m = vol.GJRGARCH().fit(pd.Series(r))
    assert m.params["gamma"] > 0.04
    assert m.persistence == pytest.approx(m.params["alpha"] + m.params["beta"] + 0.5 * m.params["gamma"])


def test_garch_t_estimates_dof():
    r, _ = synthetic.garch_series(4000, nu=5.0, seed=9)
    m = vol.GARCH(dist="t").fit(pd.Series(r))
    # nu is weakly identified: the likelihood is flat above ~6, so allow a wide band
    assert 3.5 < m.params["nu"] < 10.0
    # but the t fit must beat the normal fit on fat-tailed data
    assert m.loglik > vol.GARCH(dist="normal").fit(pd.Series(r)).loglik


def test_garch_forecast_mean_reverts_to_unconditional():
    r, _ = synthetic.garch_series(3000, seed=10)
    m = vol.GARCH().fit(pd.Series(r))
    f = m.forecast(500)
    uncond = m.params["omega"] / (1 - m.persistence)
    assert f[-1] == pytest.approx(uncond, rel=0.05)
    assert m.forecast_cumulative_variance(10) == pytest.approx(m.forecast(10).sum())


def test_qlike_zero_for_perfect_forecast():
    x = np.array([1.0, 2.0, 3.0])
    assert vol.qlike(x, x) == pytest.approx(0.0)
    assert vol.mse(x, x) == 0.0


def test_rolling_forecast_is_out_of_sample_and_aligned():
    r, _ = synthetic.garch_series(700, seed=11)
    s = pd.Series(r, index=pd.bdate_range("2018-01-01", periods=700))
    f = vol.rolling_forecast(lambda: vol.EWMA(), s, window=500, refit_every=50)
    assert len(f) == 200
    assert f.index[0] == s.index[500]
    assert f.notna().all() and (f > 0).all()


def test_compare_models_prefers_garch_on_garch_data():
    r, _ = synthetic.garch_series(1500, omega=2e-6, alpha=0.10, beta=0.85, seed=12)
    s = pd.Series(r, index=pd.bdate_range("2016-01-01", periods=1500))
    scores, fc = vol.compare_models(s, window=750, refit_every=125)
    assert set(scores.index) == {"EWMA(0.94)", "GARCH(1,1)", "GJR-GARCH"}
    assert scores.loc["GARCH(1,1)", "qlike"] < scores.loc["EWMA(0.94)", "qlike"]


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500) ** 2
    b = a + 0.5
    out = vol.diebold_mariano(a, b)
    assert out["dm_stat"] < -3 and out["p_value"] < 0.01


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
def make_returns(n=500, k=5, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((k, k)) * 0.3
    true = L @ L.T + np.eye(k) * 0.5
    X = rng.multivariate_normal(np.zeros(k), true, size=n) * 0.01
    return pd.DataFrame(X, columns=[f"A{i}" for i in range(k)], index=pd.bdate_range("2020-01-01", periods=n))


def test_sample_cov_matches_pandas():
    R = make_returns()
    np.testing.assert_allclose(cv.sample_cov(R).values, R.cov().values)


def test_ledoit_wolf_matches_sklearn():
    from sklearn.covariance import LedoitWolf

    R = make_returns(n=300, k=8)
    S, shrink = cv.ledoit_wolf(R)
    sk = LedoitWolf(assume_centered=False).fit(R.values)
    assert shrink == pytest.approx(sk.shrinkage_, rel=1e-6)
    n = len(R)
    np.testing.assert_allclose(S.values, sk.covariance_ * n / (n - 1), rtol=1e-8)


def test_ewma_cov_psd_and_shape():
    R = make_returns()
    S = cv.ewma_cov(R)
    assert S.shape == (5, 5) and cv.is_psd(S)


def test_nearest_psd_repairs():
    bad = pd.DataFrame([[1.0, 0.99, 0.0], [0.99, 1.0, -0.99], [0.0, -0.99, 1.0]], index=list("abc"), columns=list("abc"))
    assert not cv.is_psd(bad)
    assert cv.is_psd(cv.nearest_psd(bad))


def test_portfolio_vol_annualization():
    R = make_returns()
    w = pd.Series(0.2, index=R.columns)
    S = cv.sample_cov(R)
    daily = cv.portfolio_vol(S, w, annualize_=False)
    assert cv.portfolio_vol(S, w) == pytest.approx(daily * np.sqrt(252))


def test_factor_model_uses_only_real_days_when_masked():
    R, F, true_b = synthetic.factor_panel(n=1500, n_assets=3, n_factors=2, seed=21)
    a = R.columns[0]
    # corrupt the first 900 days of asset `a` with a different beta structure
    R2 = R.copy()
    R2.iloc[:900, 0] = F.iloc[:900, 0] * 5.0 + np.random.default_rng(0).standard_normal(900) * 0.01
    mask = pd.DataFrame(True, index=R2.index, columns=R2.columns)
    mask.iloc[:900, 0] = False
    full = fm.FactorModel().fit(R2, F)
    masked = fm.FactorModel().fit(R2, F, real_mask=mask)
    # masked betas recover the true ones; full-sample ones are contaminated
    np.testing.assert_allclose(masked.betas.loc[a].values, true_b.loc[a].values, atol=0.15)
    assert abs(full.betas.loc[a, "F0"] - true_b.loc[a, "F0"]) > 1.0
    assert masked.on_real_data[a] and masked.n_real_obs[a] == 600
    # the other assets are untouched by the mask
    np.testing.assert_allclose(masked.betas.iloc[1:].values, full.betas.iloc[1:].values)
    s = masked.summary()
    assert "real_only" in s.columns and s.loc[a, "n_obs"] == 600


def test_factor_model_falls_back_when_too_few_real_days():
    R, F, _ = synthetic.factor_panel(n=600, n_assets=2, n_factors=2, seed=22)
    mask = pd.DataFrame(True, index=R.index, columns=R.columns)
    mask.iloc[:500, 0] = False  # only 100 real days
    m = fm.FactorModel().fit(R, F, real_mask=mask)
    assert not m.on_real_data.iloc[0] and m.n_real_obs.iloc[0] == 100


def test_default_factors_include_semis_not_rates():
    assert "SEMIS" in fm.DEFAULT_FACTORS and "RATES" not in fm.DEFAULT_FACTORS
    assert "RATES" in fm.OPTIONAL_FACTORS
    assert "SMH" in fm.factor_tickers()

def test_gjr_does_not_stall_at_starting_values():
    """The MLE must move away from its starting values (γ recovered in both directions)."""
    r, _ = synthetic.garch_series(1700, omega=8e-6, alpha=0.10, beta=0.88, gamma=0.0, seed=4)
    m = vol.GJRGARCH().fit(pd.Series(r))
    assert m.params["gamma"] < 0.03  # true γ = 0
    r2, _ = synthetic.garch_series(1700, omega=8e-6, alpha=0.04, beta=0.88, gamma=0.10, seed=3)
    m2 = vol.GJRGARCH().fit(pd.Series(r2))
    assert m2.params["gamma"] > 2 * m2.params["alpha"]  # true γ/α = 2.5
    assert abs(m2.params["gamma"] - m2.params["alpha"]) > 0.02

def test_garch_recursion_matches_loop():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(300) * 0.01
    got = vol._garch_recursion(e, 2e-6, 0.05, 0.9, 0.08, 1e-4)
    s2 = np.empty(300)
    s2[0] = 1e-4
    for t in range(1, 300):
        s2[t] = 2e-6 + (0.05 + (0.08 if e[t - 1] < 0 else 0)) * e[t - 1] ** 2 + 0.9 * s2[t - 1]
    np.testing.assert_allclose(got, s2, rtol=1e-12)
