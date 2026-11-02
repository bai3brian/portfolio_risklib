"""Tests for risklib.data and risklib.portfolio (synthetic data, no network)."""

import numpy as np
import pandas as pd
import pytest

from risklib import data
from risklib import portfolio as pf
from risklib import synthetic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_prices() -> pd.DataFrame:
    """Three tickers over 10 business days with deliberate problems.

    AAA: complete history.
    BBB: one 2-day gap in the middle (days 4-5).
    CCC: starts late (first 3 days missing), like a recent IPO.
    """
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 101, 103, 104, 105, 104, 106, 107],
            "BBB": [50, 51, 52, 53, np.nan, np.nan, 54, 55, 56, 57],
            "CCC": [np.nan, np.nan, np.nan, 20, 21, 22, 21, 23, 24, 25],
        },
        index=dates,
        dtype=float,
    )
    return prices


# ---------------------------------------------------------------------------
# validate_prices
# ---------------------------------------------------------------------------
def test_validate_sorts_and_names_axes():
    prices = make_prices().iloc[::-1]  # reversed order
    out = data.validate_prices(prices)
    assert out.index.is_monotonic_increasing
    assert out.index.name == "date"
    assert out.columns.name == "ticker"


def test_validate_does_not_mutate_input():
    prices = make_prices().iloc[::-1]
    data.validate_prices(prices)
    assert prices.index.name is None


def test_validate_rejects_non_datetime_index():
    prices = make_prices().reset_index(drop=True)
    with pytest.raises(TypeError):
        data.validate_prices(prices)


def test_validate_rejects_duplicate_dates():
    prices = make_prices()
    prices = pd.concat([prices, prices.iloc[[0]]])
    with pytest.raises(ValueError):
        data.validate_prices(prices)


def test_validate_rejects_non_positive_prices():
    prices = make_prices()
    prices.iloc[2, 0] = 0.0
    with pytest.raises(ValueError):
        data.validate_prices(prices)


# ---------------------------------------------------------------------------
# fill_gaps
# ---------------------------------------------------------------------------
def test_fill_gaps_fills_short_gap_with_last_price():
    out = data.fill_gaps(make_prices(), max_gap=5)
    assert out["BBB"].iloc[4] == 53
    assert out["BBB"].iloc[5] == 53


def test_fill_gaps_respects_limit():
    out = data.fill_gaps(make_prices(), max_gap=1)
    assert out["BBB"].iloc[4] == 53  # first missing day filled
    assert np.isnan(out["BBB"].iloc[5])  # second one left alone


def test_fill_gaps_never_fills_before_first_price():
    out = data.fill_gaps(make_prices(), max_gap=5)
    assert out["CCC"].iloc[:3].isna().all()  # no look-ahead backfill


# ---------------------------------------------------------------------------
# drop_short_history
# ---------------------------------------------------------------------------
def test_drop_short_history_removes_thin_tickers():
    out = data.drop_short_history(make_prices(), min_obs=8)
    assert list(out.columns) == ["AAA", "BBB"]  # CCC has 7 obs, BBB has 8


def test_drop_short_history_keeps_all_when_threshold_low():
    out = data.drop_short_history(make_prices(), min_obs=1)
    assert list(out.columns) == ["AAA", "BBB", "CCC"]


# ---------------------------------------------------------------------------
# common_window
# ---------------------------------------------------------------------------
def test_common_window_is_rectangular():
    out = data.common_window(make_prices())
    assert not out.isna().any().any()
    assert len(out) == 5  # days 3,6,7,8,9 (0-indexed) have all three


# ---------------------------------------------------------------------------
# to_returns
# ---------------------------------------------------------------------------
def test_log_returns_values():
    prices = make_prices()[["AAA"]]
    out = data.to_returns(prices, kind="log")
    assert len(out) == 9  # first row has no previous price
    assert out["AAA"].iloc[0] == pytest.approx(np.log(101 / 100))


def test_log_returns_are_additive():
    prices = make_prices()[["AAA"]]
    out = data.to_returns(prices, kind="log")
    total = np.log(prices["AAA"].iloc[-1] / prices["AAA"].iloc[0])
    assert out["AAA"].sum() == pytest.approx(total)


def test_simple_returns_values():
    prices = make_prices()[["AAA"]]
    out = data.to_returns(prices, kind="simple")
    assert out["AAA"].iloc[0] == pytest.approx(0.01)


def test_returns_do_not_bridge_gaps():
    # With the gap left unfilled, the return on the day after the gap
    # must be NaN, not a return computed across the hole.
    out = data.to_returns(make_prices(), kind="log")
    assert np.isnan(out["BBB"].loc["2024-01-09"])


def test_returns_reject_bad_kind():
    with pytest.raises(ValueError):
        data.to_returns(make_prices(), kind="banana")


# ---------------------------------------------------------------------------
# normalize_weights
# ---------------------------------------------------------------------------
def test_weights_sum_to_one_and_follow_ticker_order():
    w = data.normalize_weights({"BBB": 3, "AAA": 1}, tickers=["AAA", "BBB"])
    assert list(w.index) == ["AAA", "BBB"]
    assert w.sum() == pytest.approx(1.0)
    assert w["BBB"] == pytest.approx(0.75)
    assert w.name == "weight"


def test_weights_ignore_extra_tickers():
    w = data.normalize_weights({"AAA": 1, "BBB": 1, "ZZZ": 5}, tickers=["AAA", "BBB"])
    assert list(w.index) == ["AAA", "BBB"]
    assert w.sum() == pytest.approx(1.0)


def test_weights_missing_ticker_raises():
    with pytest.raises(KeyError):
        data.normalize_weights({"AAA": 1}, tickers=["AAA", "BBB"])


def test_weights_negative_raises():
    with pytest.raises(ValueError):
        data.normalize_weights({"AAA": 1, "BBB": -0.5}, tickers=["AAA", "BBB"])


def test_weights_zero_total_raises():
    with pytest.raises(ValueError):
        data.normalize_weights({"AAA": 0, "BBB": 0}, tickers=["AAA", "BBB"])


# ---------------------------------------------------------------------------
# coverage_summary
# ---------------------------------------------------------------------------
def test_coverage_summary_contents():
    out = data.coverage_summary(make_prices())
    assert list(out.columns) == ["first_date", "last_date", "n_obs", "pct_missing"]
    assert out.index.name == "ticker"
    assert out.loc["CCC", "first_date"] == pd.Timestamp("2024-01-04")
    assert out.loc["BBB", "n_obs"] == 8
    assert out.loc["AAA", "pct_missing"] == pytest.approx(0.0)
    assert out.loc["CCC", "pct_missing"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# build_panel (the pipeline)
# ---------------------------------------------------------------------------
def test_build_panel_end_to_end():
    prices, returns = data.build_panel(make_prices(), max_gap=5, min_obs=5)
    assert list(prices.columns) == ["AAA", "BBB", "CCC"]
    assert not prices.isna().any().any()
    assert not returns.isna().any().any()
    assert len(returns) == len(prices) - 1
    assert prices.index[0] == pd.Timestamp("2024-01-04")  # CCC's first day


def test_build_panel_without_common_window_keeps_ragged_edges():
    prices, _ = data.build_panel(make_prices(), max_gap=5, min_obs=5, common=False)
    assert len(prices) == 10
    assert prices["CCC"].iloc[:3].isna().all()


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
def make_market():
    return synthetic.synthetic_market(n=600, n_assets=4, seed=1)


def test_holdings_dollars_and_weights_equivalent():
    w1 = pf.holdings_to_weights({"A": 300, "B": 700})
    w2 = pf.holdings_to_weights({"A": {"weight": 0.3}, "B": {"weight": 0.7}})
    pd.testing.assert_series_equal(w1, w2)
    assert w1["B"] == pytest.approx(0.7)


def test_holdings_shares_uses_last_price():
    last = pd.Series({"A": 10.0, "B": 20.0})
    w = pf.holdings_to_weights({"A": {"shares": 10}, "B": {"shares": 5}}, last)
    assert w["A"] == pytest.approx(0.5)


def test_holdings_shares_without_prices_raises():
    with pytest.raises(ValueError):
        pf.holdings_to_weights({"A": {"shares": 10}})


def test_daily_rebalanced_simple_return_is_weighted_sum():
    m = make_market()
    port = pf.Portfolio.from_prices(m["prices"], m["weights"].to_dict(), min_obs=100)
    simple = np.exp(port.returns) - 1
    expected = simple @ port.weights.values
    got = port.portfolio_returns(kind="simple", rebalance="daily")
    np.testing.assert_allclose(got.values, expected.values)


def test_buy_and_hold_matches_wealth_path():
    m = make_market()
    port = pf.Portfolio.from_prices(m["prices"], m["weights"].to_dict(), min_obs=100)
    r = port.portfolio_returns(kind="log", rebalance="none")
    # terminal wealth of buy-and-hold = sum_i w_i * P_iT / P_i0
    p = port.prices
    wealth = float((p.iloc[-1] / p.iloc[0]) @ port.weights.values)
    assert np.exp(r.sum()) == pytest.approx(wealth)


def test_performance_summary_on_constant_return():
    # a constant 0.04% daily log return -> zero vol, known CAGR
    r = pd.Series(0.0004, index=pd.bdate_range("2020-01-01", periods=504))
    s = pf.performance_summary(r, kind="log")
    assert s["ann_return"] == pytest.approx(np.exp(0.0004 * 252) - 1)
    assert s["ann_vol"] == pytest.approx(0.0, abs=1e-12)
    assert s["max_drawdown"] == pytest.approx(0.0)
    assert s["hit_rate"] == 1.0


def test_max_drawdown_known_value():
    # up 10%, down 20%, up 5% -> peak 1.1, trough 0.88 -> dd = -20%
    r = pd.Series([0.10, -0.20, 0.05], index=pd.bdate_range("2020-01-01", periods=3))
    assert pf.max_drawdown(r, kind="simple") == pytest.approx(-0.20)


def test_from_prices_rejects_missing_ticker():
    m = make_market()
    with pytest.raises(KeyError):
        pf.Portfolio.from_prices(m["prices"], {"NOPE": 1.0})


def test_from_prices_reports_dropped_short_history():
    m = make_market()
    prices = m["prices"].copy()
    prices.iloc[:-50, 0] = np.nan  # first asset only has 50 obs
    with pytest.raises(ValueError):
        pf.Portfolio.from_prices(prices, m["weights"].to_dict(), min_obs=100)


def test_backfill_with_proxy_matches_proxy_returns_and_keeps_real_data():
    from risklib.data import backfill_with_proxy

    idx = pd.bdate_range("2020-01-01", periods=8)
    proxy = pd.Series([10, 11, 12, 11, 13, 14, 15, 16], index=idx, dtype=float)
    prices = pd.DataFrame({"NEW": [np.nan] * 4 + [50, 52, 51, 55], "OLD": np.arange(8) + 100.0}, index=idx)
    out, first = backfill_with_proxy(prices, "NEW", proxy)
    assert first == idx[4]
    # real data untouched
    np.testing.assert_allclose(out["NEW"].iloc[4:], [50, 52, 51, 55])
    # backfilled log returns equal the proxy's log returns
    lr_new = np.log(out["NEW"]).diff().iloc[1:5]
    lr_proxy = np.log(proxy).diff().iloc[1:5]
    np.testing.assert_allclose(lr_new.values, lr_proxy.values)
    # OLD column untouched, input not mutated
    assert prices["NEW"].isna().sum() == 4
    assert out.loc[idx[4], "NEW"] == 50


def test_backfill_no_op_when_full_history():
    from risklib.data import backfill_with_proxy

    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": np.arange(5) + 1.0}, index=idx)
    out, first = backfill_with_proxy(prices, "A", prices["A"] * 2)
    assert first is None
    pd.testing.assert_frame_equal(out, prices)


def test_portfolio_with_proxy_keeps_full_window():
    m = make_market()
    prices = m["prices"].copy()
    young = prices.columns[0]
    proxy = prices.columns[1]
    prices.iloc[:400, 0] = np.nan  # young ticker: only the last 200 days are real
    holdings = {t: {"weight": float(w)} for t, w in m["weights"].items()}
    holdings[young]["proxy"] = proxy
    port = pf.Portfolio.from_prices(prices, holdings, min_obs=100)
    assert len(port.prices) == len(prices)  # nothing truncated
    tab = port.proxy_table()
    assert list(tab.index) == [young] and tab.loc[young, "proxy"] == proxy
    assert tab.loc[young, "n_backfilled"] == 400
    # without a proxy the same panel is truncated
    holdings[young].pop("proxy")
    port2 = pf.Portfolio.from_prices(prices, holdings, min_obs=100)
    assert len(port2.prices) == len(prices) - 400


def test_missing_proxy_column_raises():
    m = make_market()
    holdings = {t: {"weight": float(w)} for t, w in m["weights"].items()}
    holdings[m["prices"].columns[0]]["proxy"] = "NOPE"
    with pytest.raises(KeyError):
        pf.Portfolio.from_prices(m["prices"], holdings, min_obs=100)


def test_two_holdings_can_share_one_proxy():
    m = make_market()
    prices = m["prices"].copy()
    a, b, proxy = prices.columns[0], prices.columns[1], prices.columns[2]
    prices.iloc[:300, 0] = np.nan
    prices.iloc[:200, 1] = np.nan
    holdings = {t: {"weight": float(w)} for t, w in m["weights"].items()}
    holdings[a]["proxy"] = proxy
    holdings[b]["proxy"] = proxy
    port = pf.Portfolio.from_prices(prices, holdings, min_obs=100)
    assert len(port.prices) == len(prices)
    assert set(port.proxy_table().index) == {a, b}


def test_real_mask_marks_proxied_days():
    m = make_market()
    prices = m["prices"].copy()
    young, proxy = prices.columns[0], prices.columns[1]
    prices.iloc[:300, 0] = np.nan
    holdings = {t: {"weight": float(w)} for t, w in m["weights"].items()}
    holdings[young]["proxy"] = proxy
    port = pf.Portfolio.from_prices(prices, holdings, min_obs=100)
    mask = port.real_mask()
    assert mask.shape == port.returns.shape
    assert (~mask[young]).sum() == 299  # 300 price rows -> 299 return rows before real_from
    assert mask.drop(columns=[young]).all().all()



def test_holdings_value_and_sectors():
    last = pd.Series({"A": 10.0, "B": 20.0})
    assert pf.holdings_value({"A": {"shares": 3}, "B": {"dollars": 40}}, last) == pytest.approx(70.0)
    assert pf.holdings_value({"A": {"weight": 0.5}, "B": 40}, last) is None
    assert pf.holdings_sectors({"A": {"shares": 1, "sector": "Tech"}, "B": 1}) == {"A": "Tech"}


def test_concentration_and_sector_table():
    w = pd.Series({"A": 0.5, "B": 0.25, "C": 0.25})
    c = pf.concentration(w)
    assert c["hhi_weight"] == pytest.approx(0.375) and c["effective_n_weight"] == pytest.approx(1 / 0.375)
    risk_pct = pd.Series({"A": 80.0, "B": 10.0, "C": 10.0})
    c2 = pf.concentration(w, risk_pct)
    assert c2["effective_n_risk"] < c2["effective_n_weight"]
    st = pf.sector_table(w, {"A": "Tech", "B": "Tech"}, risk_pct)
    assert st.loc["Tech", "weight_%"] == pytest.approx(75.0) and st.loc["Unknown", "n_holdings"] == 1
    assert st.loc["Tech", "risk_%"] == pytest.approx(90.0)


def test_trailing_performance_windows():
    r = pd.Series(np.random.default_rng(0).standard_normal(1000) * 0.01, index=pd.bdate_range("2019-01-01", periods=1000))
    t = pf.trailing_performance(r)
    assert list(t.index) == ["1y", "3y", "full"]  # 5y (1260) > sample -> skipped
    assert t.loc["1y", "n_obs"] == 252 and t.loc["full", "n_obs"] == 1000


def test_portfolio_meta_value_from_shares():
    m = make_market()
    prices = m["prices"]
    holdings = {t: {"shares": 2} for t in prices.columns}
    port = pf.Portfolio.from_prices(prices, holdings, min_obs=100)
    assert port.meta["value"] == pytest.approx(float(2 * prices.iloc[-1].sum()))
