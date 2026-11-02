"""Tests for VaR/ES, Monte Carlo, stress testing and backtesting (synthetic data)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risklib.data import synthetic
from risklib.models import backtest as bt
from risklib.models import factors as fm
from risklib.models import risk, stress
from risklib.preprocessing.panel import build_panel
from risklib.preprocessing.portfolio import Portfolio


def market():
    m = synthetic.synthetic_market(n=1500, n_assets=5, seed=2)
    port = Portfolio.from_prices(m["prices"], m["weights"].to_dict(), min_obs=100)
    return m, port


# ---------------------------------------------------------------------------
# VaR / ES basics
# ---------------------------------------------------------------------------
def test_parametric_normal_matches_formula():
    r = pd.Series(np.random.default_rng(0).standard_normal(2000) * 0.01)
    var, es = risk.parametric_normal(r, 0.99, horizon=1)
    s = r.std(ddof=1)
    assert var == pytest.approx(stats.norm.ppf(0.99) * s)
    assert es == pytest.approx(s * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01)
    var10, _ = risk.parametric_normal(r, 0.99, horizon=10)
    assert var10 == pytest.approx(var * np.sqrt(10))


def test_es_exceeds_var_for_every_method():
    m, port = market()
    ps = port.portfolio_returns("simple")
    for fn in (lambda: risk.parametric_normal(ps, 0.99), lambda: risk.parametric_t(ps, 0.99)[:2], lambda: risk.historical(ps, 0.99), lambda: risk.historical(ps, 0.99, decay=0.99)):
        var, es = fn()
        assert es > var > 0


def test_parametric_t_es_closed_form_matches_numeric():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_t(4, 20000) * 0.01)
    var, es, nu = risk.parametric_t(r, 0.99)
    # numeric ES from a huge sample of the fitted distribution
    _, loc, scale = stats.t.fit(r.values, floc=0.0)
    sim = -stats.t.rvs(nu, loc=0, scale=scale, size=2_000_000, random_state=1)
    es_num = sim[sim >= np.quantile(sim, 0.99)].mean()
    assert es == pytest.approx(es_num, rel=0.03)


def test_historical_var_is_empirical_quantile():
    r = pd.Series(np.arange(1, 101) / 1000.0 - 0.05)  # -0.049 .. 0.05
    var, es = risk.historical(r, 0.95)
    assert var == pytest.approx(np.quantile(-r.values, 0.95))


def test_age_weighted_reacts_to_recent_losses():
    rng = np.random.default_rng(3)
    calm = rng.standard_normal(900) * 0.005
    wild = rng.standard_normal(100) * 0.03
    r = pd.Series(np.concatenate([calm, wild]))
    v_plain, _ = risk.historical(r, 0.99)
    v_weighted, _ = risk.historical(r, 0.99, decay=0.97)
    assert v_weighted > v_plain


def test_filtered_historical_scales_with_current_vol():
    r, _ = synthetic.garch_series(2000, omega=2e-6, alpha=0.10, beta=0.85, seed=5)
    s = pd.Series(r)
    var, es, model = risk.filtered_historical(s, 0.99, horizon=1)
    sig = np.sqrt(model.forecast(1)[0])
    assert 1.5 * sig < var < 4.5 * sig
    assert es > var


def test_monte_carlo_normal_agrees_with_parametric():
    m, port = market()
    ps = port.portfolio_returns("simple")
    var_p, _ = risk.parametric_normal(ps, 0.99)
    var_mc, es_mc, pnl = risk.monte_carlo(port.returns, port.weights, 0.99, n_sims=200_000)
    assert var_mc == pytest.approx(var_p, rel=0.05)
    assert pnl.shape == (200_000,)


def test_monte_carlo_fat_tails_raise_99_var():
    m, port = market()
    v_n, _, _ = risk.monte_carlo(port.returns, port.weights, 0.99, dist="normal", n_sims=100_000)
    v_t, _, _ = risk.monte_carlo(port.returns, port.weights, 0.99, dist="t", nu=3.5, n_sims=100_000)
    assert v_t > v_n


def test_copula_engines_keep_marginal_shape():
    m, port = market()
    sim = risk.simulate(port.returns, n_sims=50_000, horizon=1, dist="t_copula", nu=4)
    assert sim.shape == (50_000, 1, 5)
    # empirical marginals: simulated std close to historical std
    np.testing.assert_allclose(sim[:, 0, :].std(axis=0), port.returns.std().values, rtol=0.1)
    g = risk.simulate(port.returns, n_sims=50_000, horizon=1, dist="gaussian_copula")
    np.testing.assert_allclose(g[:, 0, :].std(axis=0), port.returns.std().values, rtol=0.1)


def test_simulate_paths_shape_and_start():
    m, port = market()
    paths = risk.simulate_paths(port.returns, port.weights, n_sims=100, horizon=21)
    assert paths.shape == (100, 22)
    assert (paths[:, 0] == 1.0).all()


def test_horizon_scaling_of_mc():
    m, port = market()
    v1, _, _ = risk.monte_carlo(port.returns, port.weights, 0.99, horizon=1, n_sims=50_000)
    v10, _, _ = risk.monte_carlo(port.returns, port.weights, 0.99, horizon=10, n_sims=50_000)
    assert v10 == pytest.approx(v1 * np.sqrt(10), rel=0.15)


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def test_component_var_sums_to_portfolio_var():
    m, port = market()
    cov = port.returns.cov()
    c = risk.component_var_parametric(cov, port.weights, 0.99)
    w = port.weights.values
    sig = np.sqrt(w @ cov.values @ w)
    assert c["component_var"].sum() == pytest.approx(stats.norm.ppf(0.99) * sig)
    assert c["pct_of_var"].sum() == pytest.approx(100.0)
    assert c.attrs["diversification_benefit"] > 0


def test_risk_table_shape():
    m, port = market()
    t = risk.RiskTable(alphas=(0.99,), horizons=(1,), methods=risk.ALL_METHODS, n_sims=2000).run(port.portfolio_returns("simple"), port.returns, port.weights, portfolio_value=100_000)
    assert len(t) == 8
    d = risk.RiskTable(alphas=(0.99, 0.975), horizons=(1, 10), n_sims=2000, n_boot=40).run(port.portfolio_returns("simple"), port.returns, port.weights)
    assert len(d) == 16 and {i[0].split(" (")[0] for i in d.index} == {"Parametric normal", "Historical", "Filtered historical", "MC t-copula + empirical"}
    with pytest.raises(ValueError):
        risk.RiskTable(methods=("bogus",)).run(port.portfolio_returns("simple"), port.returns, port.weights)
    assert (t["ES"] >= t["VaR"]).all()
    assert (t["VaR_$"] > 0).all()


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------
def test_historical_replay_direct_and_proxied():
    m, port = market()
    prices = m["prices"]
    scen = {"first month": (str(prices.index[0].date()), str(prices.index[21].date()))}
    summary, contrib = stress.historical_replay(prices, port.weights, scen)
    expected = float(((prices.iloc[21] / prices.iloc[0] - 1) * port.weights).sum())
    assert summary.loc["first month", "portfolio_return"] == pytest.approx(expected)
    assert summary.loc["first month", "direct"] == 5
    # an asset with no data in the window gets proxied through betas
    fp = m["factor_prices"]
    fspec = {"MKT": "MKT", "SIZE": "SIZE", "VALUE": "VALUE"}
    prices2 = prices.copy()
    prices2.iloc[:100, 0] = np.nan
    s2, c2 = stress.historical_replay(prices2, port.weights, scen, factor_prices=fp, factor_spec=fspec, betas=m["true_betas"])
    assert s2.loc["first month", "direct"] == 4 and s2.loc["first month", "proxied"] == 1


def test_historical_replay_backfilled_and_partial_factors():
    m, port = market()
    prices, fp = m["prices"], m["factor_prices"]
    scen = {"first month": (str(prices.index[0].date()), str(prices.index[21].date()))}
    fspec = {"MKT": "MKT", "SIZE": "SIZE", "VALUE": "VALUE"}
    t0, t1 = prices.columns[:2]
    # t0 counts as backfilled when its real history starts after the window
    s, _ = stress.historical_replay(prices, port.weights, scen, real_from={t0: prices.index[200]})
    assert s.loc["first month", "backfilled"] == 1 and s.loc["first month", "direct"] == 4
    # a factor whose ETF did not trade yet is dropped from the mapping and reported
    fp2 = fp.copy()
    fp2.iloc[:100, fp2.columns.get_loc("VALUE")] = np.nan
    prices2 = prices.copy()
    prices2.iloc[:100, 1] = np.nan
    s2, c2 = stress.historical_replay(prices2, port.weights, scen, factor_prices=fp2, factor_spec=fspec, betas=m["true_betas"])
    assert s2.loc["first month", "proxied"] == 1 and s2.loc["first month", "factors_missing"] == "VALUE"
    f = fp.iloc[21] / fp.iloc[0] - 1
    b = m["true_betas"].loc[t1]
    assert c2.loc["first month", t1] == pytest.approx(port.weights[t1] * (b["MKT"] * f["MKT"] + b["SIZE"] * f["SIZE"]))


def test_factor_shock_uses_portfolio_beta():
    m, port = market()
    F = build_panel(m["factor_prices"], min_obs=100)[1]
    model = fm.FactorModel().fit(port.returns, F)
    s, c = stress.factor_shock(model, port.weights, {"Mkt −20%": {"MKT": -0.20}})
    beta_p = model.portfolio_exposures(port.weights)["MKT"]
    assert s.loc["Mkt −20%", "portfolio_return"] == pytest.approx(-0.20 * beta_p)
    assert c.loc["Mkt −20%"].sum() == pytest.approx(-0.20 * beta_p)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------
def test_kupiec_accepts_correct_rate_rejects_wrong():
    rng = np.random.default_rng(0)
    good = rng.random(2500) < 0.01
    bad = rng.random(2500) < 0.04
    assert bt.kupiec_pof(good, 0.99)["p_value"] > 0.05
    assert bt.kupiec_pof(bad, 0.99)["p_value"] < 0.01


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(1)
    iid = rng.random(3000) < 0.02
    clustered = np.zeros(3000, bool)
    for s in rng.choice(3000, 12, replace=False):
        clustered[s : s + 5] = True
    assert bt.christoffersen_ind(iid)["p_value"] > 0.05
    assert bt.christoffersen_ind(clustered)["p_value"] < 0.01


def test_traffic_light_zones():
    b = np.zeros(250, bool)
    assert bt.basel_traffic_light(b)["zone"] == "green"
    b[:7] = True
    assert bt.basel_traffic_light(b)["zone"] == "yellow"
    b[:12] = True
    assert bt.basel_traffic_light(b)["zone"] == "red"


def test_acerbi_szekely_zero_when_es_correct():
    rng = np.random.default_rng(2)
    T = 20000
    loss = rng.standard_normal(T)
    var = np.full(T, stats.norm.ppf(0.99))
    es = np.full(T, stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01)
    out = bt.acerbi_szekely(loss, var, es, 0.99)
    assert abs(out["Z2"]) < 0.1
    assert out["p_value"] > 0.05


def test_rolling_var_is_out_of_sample():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.standard_normal(800) * 0.01, index=pd.bdate_range("2019-01-01", periods=800))
    f = bt.rolling_var(r, "historical", 0.99, window=500)
    assert len(f) == 300 and f.index[0] == r.index[500]
    # the first forecast uses exactly r[0:500]
    v, _ = risk.historical(r.iloc[:500], 0.99)
    assert f["VaR"].iloc[0] == pytest.approx(v)


def test_backtest_table_on_garch_data():
    r, _ = synthetic.garch_series(1300, omega=2e-6, alpha=0.10, beta=0.85, seed=4)
    s = pd.Series(np.exp(r) - 1, index=pd.bdate_range("2019-01-01", periods=1300))
    table, fc = bt.backtest_table(s, methods=("normal", "historical", "ewma", "garch", "fhs"), alpha=0.99, window=750, refit_every=63, n_sims=1000)
    assert list(table.index) == ["normal", "historical", "ewma", "garch_normal", "fhs"]  # alias resolved
    assert (table["n_days"] == 550).all()
    assert table["traffic_light_250d"].isin(["green", "yellow", "red"]).all()
    assert table["traffic_light_full"].isin(["green", "yellow", "red"]).all()
    assert (table["avg_VaR_%"] > 0).all()


def test_garch_t_backtest_row_breaches_less_than_garch_normal():
    r, _ = synthetic.garch_series(1500, omega=2e-6, alpha=0.10, beta=0.85, nu=4.0, seed=15)
    s = pd.Series(np.exp(r) - 1, index=pd.bdate_range("2018-01-01", periods=1500))
    table, fc = bt.backtest_table(s, methods=("garch_normal", "garch_t"), alpha=0.99, window=750, refit_every=125, n_sims=500)
    assert table.loc["garch_t", "breaches"] <= table.loc["garch_normal", "breaches"]
    assert (fc["garch_t"]["ES"] > fc["garch_t"]["VaR"]).all()


def test_bootstrap_ci_brackets_point_estimate():
    m, port = market()
    ps = port.portfolio_returns("simple")
    b = risk.bootstrap_ci_historical(ps, 0.99, n_boot=100)
    assert b["VaR_lo"] <= b["VaR"] <= b["VaR_hi"]
    assert b["ES_lo"] <= b["ES"] <= b["ES_hi"]
    assert b["VaR_hi"] - b["VaR_lo"] > 0
    f = risk.bootstrap_ci_fhs(ps, 0.99, n_boot=30, n_sims=2000)
    assert f["VaR_lo"] <= f["VaR_hi"] and f["VaR"] > 0


def test_risk_table_layout():
    m, port = market()
    t = risk.RiskTable(alphas=(0.99,), horizons=(1, 10), methods=risk.ALL_METHODS, n_sims=2000, n_boot=40).run(port.portfolio_returns("simple"), port.returns, port.weights)
    methods_1d = [i[0] for i in t.index if i[1] == 1]
    methods_10d = [i[0] for i in t.index if i[1] == 10]
    assert "Historical age-weighted (0.99)" in methods_1d and "Historical age-weighted (0.99)" not in methods_10d
    assert any(x.startswith("Historical (√10") for x in methods_10d)
    hist1 = t.loc[("Historical", 1, 0.99)]
    assert hist1["VaR_lo"] <= hist1["VaR"] <= hist1["VaR_hi"]
    scaled = [x for x in methods_10d if x.startswith("Historical (√10")][0]
    assert t.loc[(scaled, 10, 0.99), "VaR"] == pytest.approx(hist1["VaR"] * np.sqrt(10))
    assert (t["ES"] >= t["VaR"]).all()


def test_rate_shocks_map_through_equity_factors():
    m, port = market()
    F = build_panel(m["factor_prices"], min_obs=100)[1]
    model = fm.FactorModel().fit(port.returns, F)  # factors MKT, SIZE, VALUE
    s, c = stress.factor_shock(model, port.weights)
    row = [i for i in s.index if i.startswith("Rates +100bp")][0]
    bp = model.portfolio_exposures(port.weights)
    expected = bp["MKT"] * stress.RATE_SHOCK_PER_100BP["MKT"] + bp["VALUE"] * stress.RATE_SHOCK_PER_100BP["VALUE"]
    assert s.loc[row, "portfolio_return"] == pytest.approx(expected)
    assert "SEMIS" in s.loc[[i for i in s.index if i.startswith("Semiconductor")][0], "ignored_factors"]
