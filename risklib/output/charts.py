"""Report charts (matplotlib, PNG): conditional volatility, VaR backtest,
component VaR, stress scenarios and walk-forward wealth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# chart palette
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


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
