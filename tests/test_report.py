"""Tests for the report pipeline and the PDF renderer (synthetic data, no network)."""

import json
import re
import shutil

import pytest

from risklib.config import data_start, resolve_config
from risklib.data import synthetic
from risklib.data.sources import save_prices_csv
from risklib.output import pdf as P
from risklib.pipeline import run


def _write_market(path, n, n_assets, seed):
    (path / "data").mkdir(parents=True, exist_ok=True)
    m = synthetic.synthetic_market(n=n, n_assets=n_assets, seed=seed)
    save_prices_csv(m["prices"], path / "data" / "prices.csv")
    save_prices_csv(m["factor_prices"], path / "data" / "factor_prices.csv")
    return m


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def test_resolve_config_defaults_and_validation():
    c = resolve_config({"holdings": {"A": 1}})
    assert c["risk"]["alphas"] == [0.99, 0.975]
    assert c["risk"]["methods"] == ["normal", "historical", "fhs", "mc_t_copula"]
    assert "MKT" in c["factors"]
    assert c["output"]["pdf"] is False
    c2 = resolve_config({"holdings": {"A": 1}, "risk": {"n_sims": 10}})
    assert c2["risk"]["n_sims"] == 10 and c2["risk"]["alphas"] == [0.99, 0.975]
    with pytest.raises(KeyError):
        resolve_config({})


def test_pipeline_end_to_end(tmp_path):
    m = _write_market(tmp_path, 4 * 252, 4, seed=11)
    cfg = {
        "name": "test",
        "holdings": {t: {"weight": float(w)} for t, w in m["weights"].items()},
        "factors": {"MKT": "MKT", "SIZE": "SIZE"},
        "stress": {"scenarios": {"w": ["2020-01-06", "2020-02-06"]}, "shocks": {"mkt": {"MKT": -0.1}}},
        "risk": {"n_sims": 500, "alphas": [0.99], "horizons": [1]},
        "backtest": {"window": 300, "refit_every": 100, "n_sims": 200, "methods": ["normal", "historical", "garch"]},
        "volatility": {"window": 300, "refit_every": 100},
        "optimize": {"window": 300, "rebalance_every": 126, "strategies": ["min_variance"]},
        "data": {"source": "csv"},
        "output": {"dir": str(tmp_path), "charts": True},
    }
    res = run(cfg, out_dir=tmp_path, verbose=False)
    for f in ("report.md", "metrics.json", "charts/var_backtest.png", "bt_table.csv"):
        assert (tmp_path / f).exists(), f
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["backtest"]["n_models"] == 3
    assert 0 < metrics["factor"]["portfolio_r2"] < 1
    assert res["risk_table"]["VaR"].gt(0).all()

    # the PDF renderer must understand the report the pipeline writes
    tex, d, checks = P.build_tex(tmp_path / "report.md")
    assert d["var"] is not None and d["beta"] is not None and d["bt_best"]
    assert len(checks) >= 8
    targets = set(re.findall(r"\\hyperref\[([^\]]+)\]", tex))
    assert targets and targets <= set(re.findall(r"\\label\{([^}]+)\}", tex))


def test_data_start_reaches_back_for_scenarios():
    c = resolve_config({"holdings": {"A": 1}, "start": "2020-01-01"})
    assert data_start(c) == "2000-08-02"  # 30 days before the dot-com window
    c = resolve_config({"holdings": {"A": 1}, "start": "2020-01-01", "stress": {"history_start": None}})
    assert data_start(c) == "2020-01-01"
    c = resolve_config({"holdings": {"A": 1}, "start": "2020-01-01", "stress": {"scenarios": {"x": ["2022-01-03", "2022-02-01"]}}})
    assert data_start(c) == "2020-01-01"


def test_pipeline_replays_scenarios_before_start(tmp_path):
    m = _write_market(tmp_path, 6 * 252, 4, seed=13)
    idx = m["prices"].index
    young, parent = m["prices"].columns[0], m["prices"].columns[1]
    prices = m["prices"].copy()
    prices["PARENT"] = prices[parent]
    prices.loc[: idx[3 * 252], young] = float("nan")  # a recent listing, backfilled by PARENT
    save_prices_csv(prices, tmp_path / "data" / "prices.csv")
    holdings = {t: {"weight": float(w)} for t, w in m["weights"].items()}
    holdings[young]["proxy"] = "PARENT"
    early = (str(idx[60].date()), str(idx[100].date()))
    cfg = {
        "holdings": holdings,
        "start": str(idx[2 * 252].date()),  # estimation starts two years in
        "factors": {"MKT": "MKT", "SIZE": "SIZE"},
        "stress": {"scenarios": {"early": list(early)}},
        "risk": {"n_sims": 300, "alphas": [0.99], "horizons": [1], "n_boot": 20},
        "backtest": {"window": 300, "refit_every": 150, "n_sims": 100, "methods": ["historical"]},
        "volatility": {"window": 300, "refit_every": 150},
        "optimize": {"window": 300, "rebalance_every": 126, "strategies": ["min_variance"]},
        "data": {"source": "csv"},
        "output": {"dir": str(tmp_path), "charts": False},
    }
    res = run(cfg, out_dir=tmp_path, verbose=False)
    assert res["port"].returns.index[0] >= idx[2 * 252]
    row = res["stress_hist"].loc["early"]
    assert row["direct"] == 3 and row["backfilled"] == 1 and row["missing"] == 0
    w = res["port"].weights
    pre = prices.loc[: early[0]].iloc[-1]
    post = prices.loc[: early[1]].iloc[-1]
    r = post / pre - 1
    r[young] = r["PARENT"]
    assert row["portfolio_return"] == pytest.approx(float((r[w.index] * w).sum()))


def test_pipeline_shares_and_subset_of_methods(tmp_path):
    m = _write_market(tmp_path, 3 * 252, 3, seed=12)
    cfg = {
        "holdings": {t: {"shares": 1} for t in m["prices"].columns},
        "factors": {"MKT": "MKT"},
        "stress": {"scenarios": {"w": ["2020-01-06", "2020-02-06"]}},
        "risk": {"n_sims": 300, "alphas": [0.99], "horizons": [1], "n_boot": 20},
        "backtest": {"window": 300, "refit_every": 150, "n_sims": 100, "methods": ["historical", "fhs"]},
        "volatility": {"window": 300, "refit_every": 150},
        "optimize": {"window": 300, "rebalance_every": 126, "strategies": ["risk_parity"]},
        "data": {"source": "csv"},
        "output": {"dir": str(tmp_path), "charts": False},
    }
    res = run(cfg, out_dir=tmp_path, verbose=False)
    assert res["portfolio_value"] == pytest.approx(float(res["port"].prices.iloc[-1].sum()))
    assert res["metrics"]["portfolio"]["concentration"]["effective_n_weight"] > 1


# ---------------------------------------------------------------------------
# PDF renderer on a hand-written miniature report
# ---------------------------------------------------------------------------
MINI_MD = """# Risk report: Test book

Generated 2026-09-29 01:05. Data: 2020-01-03 → 2026-09-28 (1692 trading days, 3 holdings). Value 10,000 (from holdings at last close).

## 1. Portfolio

| ticker | weight_% | sector | risk_%_(VaR) |
|---|---|---|---|
| AAA | 50.00 | Technology | 70.00 |
| BBB | 30.00 | Healthcare | 25.00 |
| CCC_X | 20.00 | Energy | 5.00 |

### Concentration

Effective number of positions by weight **2.6**, by risk **1.8** (1/Herfindahl).

### Sectors

| sector | weight_% | risk_% | n_holdings |
|---|---|---|---|
| Technology | 50.0 | 70.0 | 1 |
| Healthcare | 30.0 | 25.0 | 1 |
| Energy | 20.0 | 5.0 | 1 |

### Proxy-backfilled histories

| ticker | proxy | real_from | n_backfilled | pct_of_sample_proxied |
|---|---|---|---|---|
| CCC_X | SPY | 2024-01-02 | 1000 | 59.1 |

### Risk profile (daily-rebalanced at today's weights)

| index | daily_rebalanced | buy_and_hold |
|---|---|---|
| ann_vol | 0.2500 | 0.2600 |
| max_drawdown | -0.3000 | -0.3100 |

### Trailing-window performance

| window | ann_return | ann_vol | sharpe | max_drawdown | n_obs | start |
|---|---|---|---|---|---|---|
| 1y | 0.10 | 0.22 | 0.45 | -0.1200 | 252 | 2025-09-26 |

## 2. Factor model (OLS, Newey-West HAC, 7 lags)

Portfolio variance explained by factors: **88.0%**.

### Portfolio exposures

| factor | exposure |
|---|---|
| MKT | 1.100 |
| SEMIS | -0.200 |

### Variance decomposition (annualized)

| source | variance | vol | pct_of_total |
|---|---|---|---|
| MKT | 0.05 | 0.22 | 88.0 |
| idiosyncratic | 0.007 | 0.08 | 12.0 |

## 4. VaR / ES (fraction of portfolio value; losses positive)

| method | horizon | alpha | VaR | ES | VaR_lo | VaR_hi | ES_lo | ES_hi | VaR_$ | ES_$ |
|---|---|---|---|---|---|---|---|---|---|---|
| Parametric normal | 1 | 0.9900 | 0.0350 | 0.0400 |  |  |  |  | 350 | 400 |
| Historical | 1 | 0.9900 | 0.0400 | 0.0550 | 0.035 | 0.045 |  |  | 400 | 550 |
| Filtered historical (GARCH) | 1 | 0.9900 | 0.0450 | 0.0520 | 0.040 | 0.050 |  |  | 450 | 520 |
| Filtered historical (GARCH) | 10 | 0.9900 | 0.1200 | 0.1400 |  |  |  |  | 1200 | 1400 |

### Component VaR 99% 1-day (Euler, EWMA(0.94) conditional covariance)

| ticker | weight | marginal_var | component_var | pct_of_var | standalone_var |
|---|---|---|---|---|---|
| AAA | 0.5 | 0.06 | 0.03 | 70.0 | 0.04 |
| BBB | 0.3 | 0.03 | 0.01 | 25.0 | 0.02 |
| CCC_X | 0.2 | 0.01 | 0.002 | 5.0 | 0.01 |

## 5. Stress tests
### Historical replay

| scenario | start | end | portfolio_return | direct | proxied | missing |
|---|---|---|---|---|---|---|
| GFC | 2008-09-12 | 2008-11-20 | 0.0000 | 0 | 0 | 3 |
| COVID crash | 2020-02-19 | 2020-03-23 | -0.3000 | 3 | 0 | 0 |

### Hypothetical factor shocks

| scenario | portfolio_return | shock | ignored_factors |
|---|---|---|---|
| Equity −20% | -0.2200 | MKT -20% | |

## 6. VaR backtest (99%, 1000 out-of-sample days, window 500)

| method | n_days | breaches | expected | breach_rate_% | kupiec_p | christoffersen_p | cc_p | traffic_light_250d | traffic_light_full | es_Z2 | es_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 1000 | 20 | 10 | 2.0000 | 0.0050 | 0.1000 | 0.0060 | green | yellow | 1.2 | 0.002 |
| gjr_normal | 1000 | 11 | 10 | 1.1000 | 0.7500 | 0.6000 | 0.8000 | green | green | 0.2 | 0.300 |

Best model by coverage among those passing conditional coverage: **gjr_normal**.

## 7. Allocation reference
### Walk-forward out-of-sample (window 500, rebalance every 63 days)

| portfolio | ann_return | ann_vol | sharpe | max_drawdown | calmar | avg_turnover |
|---|---|---|---|---|---|---|
| ledoit_wolf/risk_parity | 0.12 | 0.18 | 0.66 | -0.2 | 0.6 | 0.05 |
| current | 0.15 | 0.25 | 0.60 | -0.3 | 0.5 | 0.00 |

> Numbers are made up for this test.
"""


@pytest.fixture
def md_file(tmp_path):
    p = tmp_path / "report.md"
    p.write_text(MINI_MD, encoding="utf-8")
    return p


def test_tex_escape_specials_and_unicode():
    assert P.tex_escape("a_b & 5% #1 $") == r"a\_b \& 5\% \#1 \$"
    assert P.tex_escape("2020 → 2021") == r"2020 $\to$ 2021"
    assert P.tex_escape("Equity −20%") == r"Equity $-$20\%"
    assert P.tex_escape("日本") == "??"
    assert P.inline("**bold** and `code_x`") == r"\textbf{bold} and \texttt{code\_\allowbreak{}x}"
    assert P.num_cell("-0.25") == r"$-$0.25"


def test_parse_markdown_blocks():
    blocks = P.parse_markdown(MINI_MD)
    assert {"heading", "para", "table", "quote"} <= {b.kind for b in blocks}
    t = next(b for b in blocks if b.kind == "table")
    assert t.header == ["ticker", "weight_%", "sector", "risk_%_(VaR)"]
    assert len(t.rows) == 3


def test_extract_numbers():
    d = P.extract(P.Report(MINI_MD))
    assert d["name"] == "Test book"
    assert (d["start"], d["end"], d["n_days"], d["n_assets"]) == ("2020-01-03", "2026-09-28", 1692, 3)
    assert d["value"] == 10000
    assert d["var_method"] == "FHS" and d["var"] == pytest.approx(0.045) and d["var10"] == pytest.approx(0.12)
    assert d["var_hs"] == pytest.approx(0.04)
    assert d["beta"] == pytest.approx(1.1) and d["r2"] == pytest.approx(0.88)
    assert (d["effn_w"], d["effn_r"]) == (2.6, 1.8)
    assert d["top_risk"][0][0] == "AAA" and d["top_risk"][0][2] == pytest.approx(0.70)
    assert d["proxy_heavy"] == ["CCC_X"] and d["proxy_weight"] == pytest.approx(0.20)
    assert d["worst_stress"] == ("COVID crash", -0.30)  # the GFC row has no data and is skipped
    assert d["bt_best"] == "gjr_normal"


def test_evaluate_statuses_and_threshold_override():
    d = P.extract(P.Report(MINI_MD))
    st = {c.name: c.status for c in P.evaluate(d)}
    assert st["Risk model validity"] == "green"
    assert st["Daily VaR vs limit"] == "amber"  # 4.5% of a 5% limit
    assert st["Largest risk contributor"] == "red"
    assert st["Diversification (effective positions)"] == "red"
    assert st["Market beta"] == "green"
    assert st["Worst historical stress"] == "amber"
    assert st["1-year max drawdown"] == "green"
    assert st["Volatility regime"] == "amber"  # 0.045 / 0.040
    assert st["Data quality (proxied history)"] == "amber"
    st2 = {c.name: c.status for c in P.evaluate(d, {"var_limit": {"limit": 0.10}})}
    assert st2["Daily VaR vs limit"] == "green"


def test_links_resolve_to_unique_labels():
    rep = P.Report(MINI_MD)
    labels = P.heading_labels(rep)
    assert len(set(labels.values())) == len(labels)
    for c in P.evaluate(P.extract(rep)):
        if c.link:
            assert P.resolve_link(c.link, labels), c.link


def test_build_tex_structure(md_file):
    tex, _, _ = P.build_tex(md_file)
    assert tex.startswith(r"\documentclass") and tex.rstrip().endswith(r"\end{document}")
    assert "Portfolio health snapshot" in tex and "Risk summary" in tex and r"\section*{Detail}" in tex
    assert tex.count(r"\begin{kpi}") == 8
    assert "__NAME__" not in tex and "__DATES__" not in tex


def test_render_pdf_without_compiling(md_file):
    out = P.render_pdf(md_file, compile=False, verbose=False)
    assert out.suffix == ".tex" and out.exists()


@pytest.mark.skipif(not (shutil.which("pdflatex") or shutil.which("latexmk")), reason="no LaTeX engine on PATH")
def test_render_pdf_compiles(md_file):
    out = P.render_pdf(md_file, verbose=False)
    assert out.suffix == ".pdf" and out.stat().st_size > 10_000
