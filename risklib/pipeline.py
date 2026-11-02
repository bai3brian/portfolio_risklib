"""End-to-end risk report.

:func:`run` reads a config and runs the four stages: data (yfinance or
CSV), preprocessing, models and output. It writes to ``output.dir``:

    report.md                 full report with tables and charts
    report.pdf                optional (``output.pdf``), see :mod:`risklib.output.pdf`
    metrics.json              headline numbers, machine-readable
    charts/*.png              figures
    *.csv                     the main tables
    data/prices.csv           downloaded prices, for offline re-runs
    data/factor_prices.csv
    data/sectors.json         sector labels (config or cached yfinance lookup)

Report sections: 1 portfolio (holdings, concentration, sectors, proxies, risk
profile, trailing performance), 2 factor model, 3 volatility (out-of-sample
forecast comparison, conditional volatility), 4 VaR/ES and component VaR,
5 stress tests, 6 VaR/ES backtest, 7 allocation reference (walk-forward), and
an appendix with full-sample return statistics, which apply today's weights to
the past and are therefore hindsight.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from . import TRADING_DAYS
from .config import resolve_config, scenarios
from .data.sources import fetch_data, fetch_sectors
from .models import backtest as bt
from .models import covariance as cv
from .models import factors as fm
from .models import optimize as opt
from .models import risk, stress
from .models import volatility as vol
from .models.performance import concentration, sector_table, trailing_performance
from .output.charts import make_charts
from .output.markdown import render_markdown
from .output.metrics import headline_metrics
from .preprocessing.panel import build_panel, stress_panel
from .preprocessing.portfolio import Portfolio


def run(cfg: Mapping, out_dir: str | Path | None = None, verbose: bool = True) -> dict:
    """Run the whole pipeline. Returns a dict of every result (and writes files)."""
    cfg = resolve_config(cfg)
    out_dir = Path(out_dir or cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = (lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a)) if verbose else (lambda *a: None)

    # data and portfolio
    log("fetching prices")
    raw_prices, raw_fprices = fetch_data(cfg, out_dir)
    est_start = pd.Timestamp(cfg["start"])
    port = Portfolio.from_config(cfg, raw_prices.loc[est_start:])
    sectors = fetch_sectors(cfg, port.tickers, out_dir)
    _, fproxy_ret = build_panel(raw_fprices.loc[est_start:], min_obs=cfg["panel"]["min_obs"], common=True)
    factor_ret = fm.build_factor_returns(fproxy_ret, cfg["factors"])
    port_log = port.portfolio_returns("log")
    port_simple = port.portfolio_returns("simple")
    perf = port.performance(rf_annual=cfg["rf_annual"])
    perf_bh = port.performance(rf_annual=cfg["rf_annual"], rebalance="none")
    trailing = trailing_performance(port_log, kind="log", rf_annual=cfg["rf_annual"])

    # portfolio value: computed from holdings unless overridden in the config
    value_computed = port.meta.get("value")
    value_override = cfg["portfolio_value"]
    portfolio_value = value_override or value_computed
    value_note = ""
    if value_override and value_computed and abs(value_override / value_computed - 1) > 0.05:
        value_note = f"`portfolio_value` in the config ({value_override:,.0f}) differs from the value implied by the holdings at the last close ({value_computed:,.0f}); dollar columns use the config value."
    log(f"portfolio: {port.n_assets} assets, {len(port.returns)} days ({port.returns.index[0].date()} → {port.returns.index[-1].date()}), value {portfolio_value:,.0f}" if portfolio_value else f"portfolio: {port.n_assets} assets")

    # factor model (proxied names use real days only)
    fmodel = fm.FactorModel().fit(port.returns, factor_ret, real_mask=port.real_mask())
    exposures = fmodel.portfolio_exposures(port.weights)
    decomposition = fmodel.risk_decomposition(port.weights)
    port_r2 = fmodel.portfolio_r2(port.weights)
    log(f"factor model: portfolio R² {port_r2:.2f}")

    # volatility
    garch = vol.GARCH().fit(port_log)
    cond_vol = np.sqrt(garch.conditional_variance() * TRADING_DAYS)
    vcfg = cfg["volatility"]
    vol_window = min(vcfg["window"], len(port_log) // 2)
    vol_scores, _ = vol.compare_models(port_log, window=vol_window, refit_every=vcfg["refit_every"])
    log("volatility models compared out-of-sample")

    # covariance: Ledoit-Wolf for Monte Carlo, EWMA (default) for attribution
    lw_cov, lw_shrink = cv.ledoit_wolf(port.returns)
    ewma_cov = cv.ewma_cov(port.returns)
    rcfg = cfg["risk"]
    attr_cov = ewma_cov if rcfg["attribution_cov"] == "ewma" else lw_cov
    attr_label = "EWMA(0.94) conditional" if rcfg["attribution_cov"] == "ewma" else "Ledoit-Wolf full-sample"

    # VaR / ES and attribution
    risk_table = risk.RiskTable(alphas=tuple(rcfg["alphas"]), horizons=tuple(rcfg["horizons"]), methods=tuple(rcfg["methods"]), n_sims=rcfg["n_sims"], seed=rcfg["seed"], n_boot=rcfg["n_boot"]).run(
        port_simple, port.returns, port.weights, cov=lw_cov, portfolio_value=portfolio_value)
    risk_alpha = 0.99 if 0.99 in rcfg["alphas"] else rcfg["alphas"][0]
    comp_var = risk.component_var_parametric(attr_cov, port.weights, risk_alpha)
    conc = concentration(port.weights, comp_var["pct_of_var"])
    sectors_tbl = sector_table(port.weights, sectors, comp_var["pct_of_var"])
    log("VaR/ES and attribution done")

    # stress
    scfg = cfg["stress"]
    stress_prices, real_from = stress_panel(cfg, raw_prices)
    stress_hist, stress_hist_contrib = stress.historical_replay(stress_prices, port.weights, scenarios(cfg), factor_prices=raw_fprices, factor_spec=cfg["factors"], betas=fmodel.betas, real_from=real_from)
    shocks = scfg["shocks"] or stress.HYPOTHETICAL_SHOCKS
    stress_hypo, stress_hypo_contrib = stress.factor_shock(fmodel, port.weights, shocks)
    log("stress tests done")

    # backtest
    bcfg = cfg["backtest"]
    bt_window = min(bcfg["window"], len(port_simple) // 2)
    bt_table, bt_fc = bt.backtest_table(port_simple, methods=tuple(bcfg["methods"]), alpha=bcfg["alpha"], window=bt_window, refit_every=bcfg["refit_every"], n_sims=bcfg["n_sims"])
    ok = bt_table[bt_table["cc_p"] > 0.05] if (bt_table["cc_p"] > 0.05).any() else bt_table
    bt_best = (ok["breach_rate_%"] - (1 - bcfg["alpha"]) * 100).abs().idxmin()
    log(f"backtest done; best model {bt_best}")

    # allocation reference
    ocfg = cfg["optimize"]
    opt_window = min(ocfg["window"], len(port.returns) // 2)
    wf_summary, wf_oos, wf_weights = opt.walk_forward(port.returns, port.weights, window=opt_window, rebalance_every=ocfg["rebalance_every"], strategies=tuple(ocfg["strategies"]), w_min=ocfg["w_min"], w_max=ocfg["w_max"], exp_ret=ocfg["expected_returns"], rf_annual=cfg["rf_annual"])
    log("allocation reference done")

    res = dict(
        cfg=cfg, name=port.name, port=port, port_log=port_log, port_simple=port_simple, perf=perf, perf_bh=perf_bh, trailing=trailing,
        portfolio_value=portfolio_value, value_computed=value_computed, value_note=value_note, sectors=sectors, sectors_tbl=sectors_tbl, concentration=conc,
        fmodel=fmodel, exposures=exposures, decomposition=decomposition, port_r2=port_r2,
        garch=garch, cond_vol=cond_vol, vol_scores=vol_scores,
        lw_cov=lw_cov, lw_shrink=lw_shrink, ewma_cov=ewma_cov, attr_label=attr_label,
        risk_table=risk_table, risk_alpha=risk_alpha, component_var=comp_var,
        stress_hist=stress_hist, stress_hist_contrib=stress_hist_contrib, stress_hypo=stress_hypo, stress_hypo_contrib=stress_hypo_contrib,
        bt_table=bt_table, bt_forecasts=bt_fc, bt_best=bt_best, bt_alpha=bcfg["alpha"],
        wf_summary=wf_summary, wf_oos=wf_oos, wf_weights=wf_weights,
    )
    res["charts"] = make_charts(res, out_dir / "charts") if cfg["output"]["charts"] else {}
    res["metrics"] = headline_metrics(res)
    (out_dir / "metrics.json").write_text(json.dumps(res["metrics"], indent=2, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(res), encoding="utf-8")
    for nm in ("risk_table", "bt_table", "wf_summary", "stress_hist", "stress_hypo", "vol_scores", "component_var", "sectors_tbl"):
        res[nm].to_csv(out_dir / f"{nm}.csv")
    if cfg["output"].get("pdf"):
        from .output.pdf import render_pdf

        log("rendering PDF")
        try:
            res["pdf"] = render_pdf(out_dir / "report.md", thresholds=cfg["pdf"].get("thresholds"), verbose=verbose)
        except RuntimeError as e:
            log(f"PDF failed (report.md and report.tex are still written): {e}")
    log(f"written to {out_dir.resolve()}")
    return res
