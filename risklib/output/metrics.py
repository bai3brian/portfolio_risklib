"""Headline numbers for metrics.json (and the PDF snapshot)."""

from __future__ import annotations


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
