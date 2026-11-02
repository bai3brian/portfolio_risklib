"""report.md: every table and chart, section by section."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import stress


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
    L.append("\nMin-variance and risk parity need no expected-return estimate; max-Sharpe and target-vol allocators remain in `risklib.models.optimize` but are not reported because their out-of-sample results are dominated by return-estimation error.\n")
    if "walk_forward" in ch:
        L.append(f"\n![walk_forward]({ch['walk_forward']})\n")

    L.append("\n## Appendix: full-sample return statistics (hindsight)\n")
    L.append("Today's weights applied to the whole history. Because the weights are on stocks that have already performed, these figures overstate what the portfolio could have been expected to earn; they are kept for completeness only.\n")
    app = pd.DataFrame({"daily_rebalanced": p, "buy_and_hold": res["perf_bh"]}).loc[["ann_return", "sharpe", "sortino", "calmar", "start", "end"]]
    L.append(_md(app, "{:.4f}"))
    L.append("\n---\n*Generated by risklib. Every table is reproducible from `data/*.csv` with `data.source: csv`.*\n")
    return "\n".join(L)
