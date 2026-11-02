"""Tests for risklib.models.optimize (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from risklib.data import synthetic
from risklib.models import covariance as cv
from risklib.models import optimize as opt
from risklib.preprocessing.portfolio import Portfolio


def make():
    m = synthetic.synthetic_market(n=1200, n_assets=5, seed=6)
    port = Portfolio.from_prices(m["prices"], m["weights"].to_dict(), min_obs=100)
    return port, cv.sample_cov(port.returns)


def test_min_variance_two_asset_closed_form():
    # uncorrelated assets with variances 1 and 4 -> w = (0.8, 0.2)
    S = pd.DataFrame([[1.0, 0.0], [0.0, 4.0]], index=["a", "b"], columns=["a", "b"])
    w = opt.min_variance(S)
    np.testing.assert_allclose(w.values, [0.8, 0.2], atol=1e-5)


def test_min_variance_beats_equal_weight_in_sample():
    port, S = make()
    w = opt.min_variance(S)
    eq = pd.Series(0.2, index=S.columns)
    assert cv.portfolio_vol(S, w) <= cv.portfolio_vol(S, eq)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()


def test_box_constraints_respected():
    port, S = make()
    w = opt.min_variance(S, w_min=0.05, w_max=0.4)
    assert (w >= 0.05 - 1e-8).all() and (w <= 0.4 + 1e-8).all()


def test_max_sharpe_has_highest_sharpe_among_candidates():
    port, S = make()
    mu = opt.expected_returns(port.returns, "mean")
    w = opt.max_sharpe(S, mu)
    def sharpe(x):
        return (x.values @ mu.values) / np.sqrt(x.values @ S.values @ x.values)
    assert sharpe(w) >= sharpe(opt.min_variance(S)) - 1e-9
    assert sharpe(w) >= sharpe(pd.Series(0.2, index=S.columns)) - 1e-9


def test_risk_parity_equalizes_contributions():
    port, S = make()
    w = opt.risk_parity(S)
    rc = opt.risk_contributions(S, w)["pct_of_risk"]
    np.testing.assert_allclose(rc.values, 20.0, atol=0.5)


def test_risk_parity_with_budgets_and_bounds():
    port, S = make()
    b = pd.Series([0.4, 0.15, 0.15, 0.15, 0.15], index=S.columns)
    w = opt.risk_parity(S, budgets=b)
    rc = opt.risk_contributions(S, w)["pct_of_risk"]
    np.testing.assert_allclose(rc.values, b.values * 100, atol=1.0)
    w2 = opt.risk_parity(S, w_min=0.1, w_max=0.3)
    assert (w2 >= 0.1 - 1e-6).all() and (w2 <= 0.3 + 1e-6).all()


def test_target_vol_respects_cap():
    port, S = make()
    mu = opt.expected_returns(port.returns, "mean")
    mv_vol = cv.portfolio_vol(S, opt.min_variance(S))
    cap = mv_vol * 1.2
    w = opt.target_vol(S, mu, cap)
    assert cv.portfolio_vol(S, w) <= cap + 1e-6
    # unreachable cap falls back to min variance
    w_lo = opt.target_vol(S, mu, mv_vol * 0.5)
    np.testing.assert_allclose(w_lo.values, opt.min_variance(S).values, atol=1e-6)


def test_expected_returns_shrinkage_between_mean_and_grand_mean():
    port, _ = make()
    mu = opt.expected_returns(port.returns, "mean")
    sh = opt.expected_returns(port.returns, "shrunk")
    g = mu.mean()
    assert ((sh - g).abs() <= (mu - g).abs() + 1e-15).all()
    assert (opt.expected_returns(port.returns, "zero") == 0).all()


def test_efficient_frontier_monotone():
    port, S = make()
    mu = opt.expected_returns(port.returns, "mean")
    ef = opt.efficient_frontier(S, mu, n_points=10)
    assert len(ef) == 10
    assert ef["ann_return"].is_monotonic_increasing
    assert ef["ann_vol"].iloc[0] <= ef["ann_vol"].iloc[-1] + 1e-9


def test_walk_forward_shapes_and_benchmarks():
    port, _ = make()
    summary, oos, wh = opt.walk_forward(port.returns, port.weights, window=500, rebalance_every=126, strategies=("min_variance", "risk_parity"))
    assert set(summary.index) == {"sample/min_variance", "sample/risk_parity", "ledoit_wolf/min_variance", "ledoit_wolf/risk_parity", "equal_weight", "current"}
    assert len(oos) == len(port.returns) - 500
    assert (summary["avg_turnover"] >= 0).all()
    assert summary.loc["equal_weight", "avg_turnover"] == 0.0
    assert wh.index.names == ["rebalance_date", "portfolio"]
