"""End-to-end risk report.

:func:`run` reads a config, fetches prices (yfinance or CSV), runs every
module and writes to ``output.dir``:

    report.md                 full report with tables and charts
    report.pdf                optional (``output.pdf``), see :mod:`risklib.pdfreport`
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

from . import backtest as bt
from . import covariance as cv
from . import factors as fm
from . import optimize as opt
from . import risk, stress
from . import volatility as vol
from .data import TRADING_DAYS, backfill_with_proxy, build_panel, download_prices, fill_gaps, load_prices_csv, save_prices_csv
from .portfolio import Portfolio, concentration, load_config, sector_table, trailing_performance

# chart palette
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


# ---------------------------------------------------------------------------
# config defaults
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "start": "2015-01-01",
    "end": None,
    "portfolio_value": None,  # override only; computed from shares/dollars when possible
    "rf_annual": 0.0,
    "panel": {"max_gap": 5, "min_obs": 252, "common": True},
    "factors": "DEFAULT",
    "volatility": {"window": 1000, "refit_every": 21},
    "risk": {"alphas": [0.99, 0.975], "horizons": [1, 10], "methods": list(risk.DEFAULT_METHODS), "n_sims": 20000, "seed": 0, "n_boot": 300, "attribution_cov": "ewma"},
    "backtest": {"alpha": 0.99, "window": 500, "refit_every": 21, "n_sims": 5000, "methods": list(bt.DEFAULT_METHODS)},
    "stress": {"scenarios": None, "shocks": None, "history_start": "auto"},
    "optimize": {"window": 500, "rebalance_every": 63, "w_min": 0.0, "w_max": 1.0, "strategies": ["min_variance", "risk_parity"], "expected_returns": "shrunk"},
    "data": {"source": "yfinance", "prices_csv": None, "factor_csv": None, "lookup_sectors": True},
    "output": {"dir": "reports", "charts": True, "pdf": False},
    "pdf": {"thresholds": {}},
}


def _merge(base: Mapping, override: Mapping | None) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, Mapping) and isinstance(base.get(k), Mapping) else v
    return out


def resolve_config(cfg: Mapping) -> dict:
    c = _merge(DEFAULTS, cfg)
    if "holdings" not in c:
        raise KeyError("config needs a 'holdings' section")
    if c["factors"] == "DEFAULT":
        c["factors"] = dict(fm.DEFAULT_FACTORS)
    return c


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _scenarios(cfg: dict) -> dict[str, tuple[str, str]]:
    return {k: tuple(v) for k, v in (cfg["stress"]["scenarios"] or stress.HISTORICAL_SCENARIOS).items()}


def data_start(cfg: dict) -> str:
    """First download date: ``start``, or earlier if the stress scenarios need it.

    ``stress.history_start`` is ``"auto"`` (30 days before the earliest
    scenario), a date, or null (no extra history).
    """
    hs = cfg["stress"].get("history_start", "auto")
    if hs is None:
        return cfg["start"]
    if hs == "auto":
        first = min(pd.Timestamp(s) for s, _ in _scenarios(cfg).values())
        hs = first - pd.Timedelta(days=30)
    return str(min(pd.Timestamp(hs), pd.Timestamp(cfg["start"])).date())


def fetch_data(cfg: dict, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PRICES for holdings and factor proxies from :func:`data_start`, from yfinance or CSV."""
    tickers = list(cfg["holdings"].keys())
    proxies = [s["proxy"] for s in cfg["holdings"].values() if isinstance(s, Mapping) and s.get("proxy")]
    for p in proxies:
        if p not in tickers:  # a shared proxy is added once
            tickers.append(p)
    ftick = fm.factor_tickers(cfg["factors"])
    src = cfg["data"]["source"]
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if src == "yfinance":
        first = data_start(cfg)
        prices = download_prices(tickers, first, cfg["end"])
        fprices = download_prices(ftick, first, cfg["end"])
        save_prices_csv(prices, data_dir / "prices.csv")
        save_prices_csv(fprices, data_dir / "factor_prices.csv")
    elif src == "csv":
        prices = load_prices_csv(cfg["data"]["prices_csv"] or data_dir / "prices.csv")
        fprices = load_prices_csv(cfg["data"]["factor_csv"] or data_dir / "factor_prices.csv")
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise KeyError(f"{missing} not in {cfg['data']['prices_csv'] or data_dir / 'prices.csv'}; re-run without --offline")
        prices = prices[tickers]
    else:
        raise ValueError("data.source must be 'yfinance' or 'csv'")
    return prices, fprices


def stress_panel(cfg: dict, raw_prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    """Long-history holding prices for stress replays, proxy-backfilled where configured.

    Returns the price panel and each backfilled holding's first real date.
    """
    holdings = cfg["holdings"]
    full = fill_gaps(raw_prices[list(holdings)].sort_index(), cfg["panel"]["max_gap"])
    real_from: dict[str, pd.Timestamp] = {}
    for t, spec in holdings.items():
        p = spec.get("proxy") if isinstance(spec, Mapping) else None
        if p and p in raw_prices.columns:
            try:
                full, first = backfill_with_proxy(full, t, raw_prices[p].sort_index().ffill(limit=cfg["panel"]["max_gap"]))
            except ValueError:
                continue
            if first is not None:
                real_from[t] = first
    return full, real_from


def fetch_sectors(cfg: dict, tickers: list[str], out_dir: Path) -> dict[str, str]:
    """Sector per ticker: YAML ``sector:`` first, then a cached yfinance lookup."""
    from .portfolio import holdings_sectors

    sectors = holdings_sectors(cfg["holdings"])
    cache = out_dir / "data" / "sectors.json"
    cached: dict[str, str] = {}
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
    need = [t for t in tickers if t not in sectors and t not in cached]
    if need and cfg["data"]["source"] == "yfinance" and cfg["data"].get("lookup_sectors", True):
        try:
            import yfinance as yf

            for t in need:
                try:
                    info = yf.Ticker(t).info or {}
                    cached[t] = str(info.get("sector") or "Unknown")
                except Exception:
                    cached[t] = "Unknown"
        except ImportError:
            pass
    out = {t: sectors.get(t, cached.get(t, "Unknown")) for t in tickers}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({**cached, **sectors}, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def _style(ax, title: str, ylabel: str = ""):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def make_charts(res: dict, chart_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 150})
    out = {}

    # conditional volatility: GARCH(1,1) against its long-run level
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cv_ = res["cond_vol"]
    ax.plot(cv_.index, cv_.values * 100, color=PALETTE[0], lw=1.4, label="GARCH(1,1) conditional volatility")
    lr = res["garch"].unconditional_vol_ann
    if lr == lr:
        ax.axhline(lr * 100, color=MUTED, lw=1, ls="--", label=f"long-run level {lr:.1%}")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Annualized conditional volatility", "% ann.")
    fig.tight_layout(); fig.savefig(chart_dir / "conditional_vol.png"); plt.close(fig)
    out["conditional_vol"] = "charts/conditional_vol.png"

    # VaR backtest of the selected model
    f = res["bt_forecasts"][res["bt_best"]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(f.index, -f["loss"] * 100, color=MUTED, lw=0.7, label="daily return")
    ax.plot(f.index, -f["VaR"] * 100, color=PALETTE[0], lw=1.3, label=f"−VaR {res['bt_alpha']:.0%} ({res['bt_best']})")
    br = f[f["breach"]]
    ax.scatter(br.index, -br["loss"] * 100, color=PALETTE[7], s=14, zorder=3, label=f"breaches ({len(br)})")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax, "Out-of-sample VaR backtest", "% daily")
    fig.tight_layout(); fig.savefig(chart_dir / "var_backtest.png"); plt.close(fig)
    out["var_backtest"] = "charts/var_backtest.png"

    # component VaR: risk share vs weight share
    c = res["component_var"].sort_values("pct_of_var", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(c) + 1.5))
    y = np.arange(len(c))
    ax.barh(y + 0.18, c["pct_of_var"], color=PALETTE[0], height=0.36, label="% of VaR")
    ax.barh(y - 0.18, c["weight"] * 100, color=PALETTE[1], height=0.36, label="% of weight")
    ax.set_yticks(y); ax.set_yticklabels(c.index)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, f"Risk share vs weight share ({res['risk_alpha']:.0%} conditional component VaR)", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(chart_dir / "component_var.png"); plt.close(fig)
    out["component_var"] = "charts/component_var.png"

    # stress scenarios
    s = pd.concat([res["stress_hist"]["portfolio_return"], res["stress_hypo"]["portfolio_return"]]).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(s) + 1.5))
    ax.barh(s.index, s.values * 100, color=[PALETTE[7] if v < 0 else PALETTE[2] for v in s.values], height=0.6)
    for yy, v in enumerate(s.values * 100):
        ax.text(v, yy, f" {v:.1f}%", va="center", fontsize=8, color=INK, ha="left" if v >= 0 else "right")
    _style(ax, "Stress scenarios: portfolio return at today's weights", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(chart_dir / "stress.png"); plt.close(fig)
    out["stress"] = "charts/stress.png"

    # walk-forward wealth
    oos = res["wf_oos"]
    fig, ax = plt.subplots(figsize=(9, 4))
    keep = [c_ for c_ in oos.columns if c_ in ("current", "equal_weight") or c_.startswith("ledoit_wolf/")]
    for i, c_ in enumerate(keep[:6]):
        w = np.exp(oos[c_].cumsum())
        ax.plot(w.index, w.values, color=PALETTE[i], lw=1.4, label=c_)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Walk-forward out-of-sample growth of $1", "wealth")
    fig.tight_layout(); fig.savefig(chart_dir / "walk_forward.png"); plt.close(fig)
    out["walk_forward"] = "charts/walk_forward.png"
    return out


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------
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
    stress_hist, stress_hist_contrib = stress.historical_replay(stress_prices, port.weights, _scenarios(cfg), factor_prices=raw_fprices, factor_spec=cfg["factors"], betas=fmodel.betas, real_from=real_from)
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
        from .pdfreport import render_pdf

        log("rendering PDF")
        try:
            res["pdf"] = render_pdf(out_dir / "report.md", thresholds=cfg["pdf"].get("thresholds"), verbose=verbose)
        except RuntimeError as e:
            log(f"PDF failed (report.md and report.tex are still written): {e}")
    log(f"written to {out_dir.resolve()}")
    return res


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def headline_metrics(res: dict) -> dict:
    p, bt_t, wf, rt = res["perf"], res["bt_table"], res["wf_summary"], res["risk_table"]
    a = res["risk_alpha"]

    def _row(method, h=1, col="VaR"):
        for m in rt.index.get_level_values(0).unique():
            if m.startswith(method):
                try:
                    return float(rt.loc[(m, h, a), col])
                except KeyError:
                    return None
        return None

    best = res["bt_best"]
    vs = res["vol_scores"]
    base = vs.index[0]
    best_vol = vs["qlike"].idxmin()
    conc = res["concentration"]
    g = res["garch"]
    lw_mv, s_mv, lw_rp = "ledoit_wolf/min_variance", "sample/min_variance", "ledoit_wolf/risk_parity"
    return {
        "portfolio": {"name": res["name"], "n_assets": res["port"].n_assets, "n_days": int(p["n_obs"]), "start": str(p["start"].date()), "end": str(p["end"].date()),
                      "value": res["portfolio_value"], "value_computed": res["value_computed"],
                      "proxied": {t: v["proxy"] for t, v in res["port"].meta.get("proxied", {}).items()},
                      "concentration": {k: float(v) for k, v in conc.items()},
                      "sectors_weight_%": res["sectors_tbl"]["weight_%"].round(2).to_dict(),
                      "sectors_risk_%": res["sectors_tbl"]["risk_%"].round(2).to_dict() if "risk_%" in res["sectors_tbl"] else {}},
        "risk_profile": {k: float(p[k]) for k in ("ann_vol", "max_drawdown", "skew", "kurtosis", "hit_rate")},
        "trailing": res["trailing"][["ann_return", "ann_vol", "sharpe", "max_drawdown"]].round(4).to_dict(orient="index"),
        "full_sample_returns_hindsight": {k: float(p[k]) for k in ("ann_return", "sharpe", "sortino", "calmar")},
        "factor": {"portfolio_r2": float(res["port_r2"]), "exposures": res["exposures"].round(3).to_dict(), "hac_lags": int(res["fmodel"].hac_lags),
                   "n_assets_on_real_data_only": int(res["fmodel"].on_real_data.sum())},
        "volatility": {"garch_current_vol_ann": float(res["cond_vol"].iloc[-1]), "garch_long_run_vol_ann": float(g.unconditional_vol_ann), "garch_persistence": float(g.persistence), "garch_half_life_days": float(g.half_life),
                       "oos_qlike": vs["qlike"].round(4).to_dict(), "best_oos_model": best_vol, "qlike_improvement_vs_" + base + "_%": float((1 - vs.loc[best_vol, "qlike"] / vs.loc[base, "qlike"]) * 100),
                       "dm_pvalue_best": float(vs.loc[best_vol, "dm_pvalue"]) if best_vol != base else None},
        "covariance": {"ledoit_wolf_shrinkage": float(res["lw_shrink"]), "attribution_cov": res["attr_label"]},
        "var": {"alpha": a, "parametric_normal_1d": _row("Parametric normal"), "historical_1d": _row("Historical"), "historical_1d_ci": [_row("Historical", 1, "VaR_lo"), _row("Historical", 1, "VaR_hi")],
                "fhs_1d": _row("Filtered historical"), "fhs_1d_ci": [_row("Filtered historical", 1, "VaR_lo"), _row("Filtered historical", 1, "VaR_hi")],
                "fhs_1d_es": _row("Filtered historical", 1, "ES"), "mc_t_copula_1d": _row("MC t-copula"), "fhs_10d": _row("Filtered historical", 10),
                "top_risk_contributor": res["component_var"]["pct_of_var"].idxmax(), "top_contributor_pct_of_var": float(res["component_var"]["pct_of_var"].max()),
                "top_contributor_weight_pct": float(res["component_var"]["weight"].max() * 100),
                "diversification_benefit_pct_of_standalone": float(res["component_var"].attrs["diversification_benefit"] / res["component_var"]["standalone_var"].sum() * 100)},
        "stress": {"worst_historical": res["stress_hist"]["portfolio_return"].idxmin(), "worst_historical_return": float(res["stress_hist"]["portfolio_return"].min()),
                   "historical": res["stress_hist"]["portfolio_return"].round(4).to_dict(), "hypothetical": res["stress_hypo"]["portfolio_return"].round(4).to_dict()},
        "backtest": {"alpha": res["bt_alpha"], "n_days": int(bt_t["n_days"].iloc[0]), "best_model": best, "best_breach_rate_%": float(bt_t.loc[best, "breach_rate_%"]), "best_kupiec_p": float(bt_t.loc[best, "kupiec_p"]), "best_christoffersen_p": float(bt_t.loc[best, "christoffersen_p"]), "best_cc_p": float(bt_t.loc[best, "cc_p"]),
                     "best_traffic_light_full": bt_t.loc[best, "traffic_light_full"], "best_es_Z2": float(bt_t.loc[best, "es_Z2"]), "best_es_p": float(bt_t.loc[best, "es_p"]),
                     "n_models_passing_cc": int((bt_t["cc_p"] > 0.05).sum()), "n_models": int(len(bt_t)),
                     "normal_breach_rate_%": float(bt_t.loc["normal", "breach_rate_%"]) if "normal" in bt_t.index else None,
                     "garch_normal_breach_rate_%": float(bt_t.loc["garch_normal", "breach_rate_%"]) if "garch_normal" in bt_t.index else None,
                     "garch_t_breach_rate_%": float(bt_t.loc["garch_t", "breach_rate_%"]) if "garch_t" in bt_t.index else None},
        "allocation": {"walk_forward": wf.round(4).to_dict(orient="index"),
                       "lw_vs_sample_minvar_vol_reduction_%": float((1 - wf.loc[lw_mv, "ann_vol"] / wf.loc[s_mv, "ann_vol"]) * 100) if lw_mv in wf.index and s_mv in wf.index else None,
                       "risk_parity_vs_current": {"sharpe": [float(wf.loc[lw_rp, "sharpe"]), float(wf.loc["current", "sharpe"])], "ann_vol": [float(wf.loc[lw_rp, "ann_vol"]), float(wf.loc["current", "ann_vol"])], "turnover": float(wf.loc[lw_rp, "avg_turnover"])} if lw_rp in wf.index else None},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    d = df.copy().reset_index()

    def _fmt(x):
        if isinstance(x, (float, np.floating)):
            return "" if pd.isna(x) else floatfmt.format(x)
        if isinstance(x, pd.Timestamp):
            return str(x.date())
        if isinstance(x, (bool, np.bool_)):
            return "yes" if x else "no"
        return str(x)

    for c in d.columns:
        d[c] = d[c].map(_fmt)
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def render_markdown(res: dict) -> str:
    m, p, ch, cfg = res["metrics"], res["perf"], res["charts"], res["cfg"]
    L = [f"# Risk report: {res['name']}\n"]
    val = f"Value {res['portfolio_value']:,.0f}" + (" (from holdings at last close)" if not cfg["portfolio_value"] else " (config override)") if res["portfolio_value"] else "Value n/a (weights only)"
    L.append(f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Data: {m['portfolio']['start']} → {m['portfolio']['end']} ({m['portfolio']['n_days']} trading days, {m['portfolio']['n_assets']} holdings). {val}. Source: `{cfg['data']['source']}`.\n")
    if res["value_note"]:
        L.append(f"> {res['value_note']}\n")

    L.append("## 1. Portfolio\n")
    wt = res["port"].weights.to_frame().assign(weight_pct=lambda d: d["weight"] * 100).drop(columns="weight")
    wt["sector"] = pd.Series(res["sectors"])
    wt["pct_of_var"] = res["component_var"]["pct_of_var"]
    L.append(_md(wt.rename(columns={"weight_pct": "weight_%", "pct_of_var": "risk_% (VaR)"}), "{:.2f}"))
    c = res["concentration"]
    L.append(f"\n### Concentration\n\nEffective number of positions by weight **{c['effective_n_weight']:.1f}**, by risk **{c['effective_n_risk']:.1f}** (1/Herfindahl). Top-1 holding {c['top1_weight']:.1%} of weight and {c['top1_risk']:.1%} of VaR; top-5 {c['top5_weight']:.1%} / {c['top5_risk']:.1%}.\n")
    L.append("### Sectors\n" + _md(res["sectors_tbl"], "{:.1f}"))
    pt = res["port"].proxy_table()
    if len(pt):
        L.append("\n### Proxy-backfilled histories\n\nBefore `real_from` these holdings' returns are the proxy's. Factor betas for them are estimated on real days only (section 2); covariance and stress replays use the full backfilled panel.\n")
        L.append(_md(pt, "{:.1f}"))
    L.append("\n### Risk profile (daily-rebalanced at today's weights)\n")
    rp = pd.DataFrame({"daily_rebalanced": p, "buy_and_hold": res["perf_bh"]}).loc[["ann_vol", "max_drawdown", "skew", "kurtosis", "hit_rate", "n_obs"]]
    L.append(_md(rp, "{:.4f}"))
    L.append("\n### Trailing-window performance\n\nThese are the figures to compare with a benchmark. The full-sample row, and all return statistics, describe today's weights applied to the past — hindsight, not expectation (see appendix).\n")
    L.append(_md(res["trailing"], "{:.4f}"))

    L.append("\n## 2. Factor model (OLS, Newey-West HAC, %d lags)\n" % m["factor"]["hac_lags"])
    fac_desc = ", ".join(f"{k}={v}" if isinstance(v, str) else f"{k}={v[0]}−{v[1]}" for k, v in cfg["factors"].items())
    L.append(f"Portfolio variance explained by factors: **{m['factor']['portfolio_r2']:.1%}**. Factors: {fac_desc}. Proxied holdings' betas use real data only (`real_only` column).\n")
    L.append("### Portfolio exposures\n" + _md(res["exposures"].to_frame(), "{:.3f}"))
    L.append("\n### Variance decomposition (annualized)\n" + _md(res["decomposition"], "{:.4f}"))
    betas = res["fmodel"].summary().drop(columns=["alpha_ann", "alpha_tstat"])
    L.append("\n### Per-asset betas\n" + _md(betas, "{:.3f}"))

    L.append("\n## 3. Volatility (portfolio returns)\n")
    vm = m["volatility"]
    L.append(f"GARCH(1,1) conditional volatility is **{vm['garch_current_vol_ann']:.1%}** annualized against a long-run level of {vm['garch_long_run_vol_ann']:.1%}; shocks decay with a half-life of {vm['garch_half_life_days']:.0f} days (persistence {vm['garch_persistence']:.3f}).\n")
    if "conditional_vol" in ch:
        L.append(f"\n![vol]({ch['conditional_vol']})\n")
    L.append("### Out-of-sample forecast comparison (rolling refit)\n" + _md(res["vol_scores"], "{:.4g}"))
    L.append("\nLower QLIKE is a better variance forecast; `dm_pvalue` < 0.05 means the difference from the first model is statistically significant (Diebold-Mariano).\n")

    L.append("\n## 4. VaR / ES (fraction of portfolio value; losses positive)\n")
    L.append("`VaR_lo`/`VaR_hi` are bootstrap 95% confidence intervals (historical: resampled days; FHS: resampled standardized residuals, GARCH parameters fixed). Multi-day historical rows are √h-scaled from the 1-day estimate. Monte Carlo uses the Ledoit-Wolf covariance. Methods and confidence levels are set by `risk.methods` and `risk.alphas`.\n")
    L.append(_md(res["risk_table"], "{:.4f}"))
    L.append(f"\n### Component VaR {res['risk_alpha']:.0%} 1-day (Euler, {res['attr_label']} covariance)\n" + _md(res["component_var"], "{:.4f}"))
    L.append(f"\nDiversification benefit: {m['var']['diversification_benefit_pct_of_standalone']:.1f}% of the sum of standalone VaRs. The covariance is conditional, so the attribution reflects today's volatility like the FHS headline.\n")
    if "component_var" in ch:
        L.append(f"\n![cvar]({ch['component_var']})\n")

    L.append("\n## 5. Stress tests\n### Historical replay\n" + _md(res["stress_hist"], "{:.4f}"))
    L.append("\nToday's weights applied to each window. Per holding, the window return comes from its own prices (`direct`), its proxy-backfilled history (`backfilled`), its factor betas times the factor returns over the window (`proxied`, idiosyncratic part zero), or is unavailable (`missing`, contributes 0). `factors_missing` lists factors whose ETFs did not trade in the window and so were left out of the `proxied` mapping. Price history for the replays is downloaded back to `stress.history_start`, independently of the estimation window.\n")
    L.append("\n### Hypothetical factor shocks\n" + _md(res["stress_hypo"], "{:.4f}"))
    rate_desc = ", ".join(f"{k} {v:+.0%}" for k, v in stress.RATE_SHOCK_PER_100BP.items())
    L.append(f"\nRate shocks are mapped through equities the way 2022 transmitted them (per +100bp: {rate_desc}); see `stress.RATE_SHOCK_PER_100BP`.\n")
    if "stress" in ch:
        L.append(f"\n![stress]({ch['stress']})\n")

    L.append(f"\n## 6. VaR backtest ({res['bt_alpha']:.0%}, {m['backtest']['n_days']} out-of-sample days, window {cfg['backtest']['window']})\n" + _md(res["bt_table"], "{:.4f}"))
    L.append(f"\nBest model by coverage among those passing conditional coverage: **{res['bt_best']}**. `garch_normal` uses a normal quantile on a GARCH σ, so comparing it with `fhs` (same σ, empirical tails) isolates the tail assumption. Models are set by `backtest.methods`. `es_Z2` > 0 means realized losses beyond VaR exceeded ES; `es_p` is the one-sided bootstrap probability. `traffic_light_250d` is the Basel zone on the last 250 days only; `traffic_light_full` applies the same binomial thresholds to the whole backtest.\n")
    if "var_backtest" in ch:
        L.append(f"\n![bt]({ch['var_backtest']})\n")

    L.append("\n## 7. Allocation reference\n### Walk-forward out-of-sample (window %d, rebalance every %d days)\n" % (cfg["optimize"]["window"], cfg["optimize"]["rebalance_every"]) + _md(res["wf_summary"], "{:.4f}"))
    L.append("\nMin-variance and risk parity need no expected-return estimate; max-Sharpe and target-vol allocators remain in `risklib.optimize` but are not reported because their out-of-sample results are dominated by return-estimation error.\n")
    if "walk_forward" in ch:
        L.append(f"\n![walk_forward]({ch['walk_forward']})\n")

    L.append("\n## Appendix: full-sample return statistics (hindsight)\n")
    L.append("Today's weights applied to the whole history. Because the weights are on stocks that have already performed, these figures overstate what the portfolio could have been expected to earn; they are kept for completeness only.\n")
    app = pd.DataFrame({"daily_rebalanced": p, "buy_and_hold": res["perf_bh"]}).loc[["ann_return", "sharpe", "sortino", "calmar", "start", "end"]]
    L.append(_md(app, "{:.4f}"))
    L.append("\n---\n*Generated by risklib. Every table is reproducible from `data/*.csv` with `data.source: csv`.*\n")
    return "\n".join(L)
