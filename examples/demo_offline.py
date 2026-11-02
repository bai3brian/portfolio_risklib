"""Offline demo on synthetic data.

Builds a synthetic market with ``risklib.synthetic`` and runs the full report
pipeline on it, so the project can be tried without network access. The
numbers describe the synthetic market only. For a real portfolio run
``risklib report my_portfolio.yaml``, which downloads prices from yfinance.

Usage:  python examples/demo_offline.py [output_dir] [--pdf]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risklib import synthetic  # noqa: E402
from risklib.data import save_prices_csv  # noqa: E402
from risklib.report import run  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0] if args else "reports_demo_synthetic")
(out / "data").mkdir(parents=True, exist_ok=True)

m = synthetic.synthetic_market(n=8 * 252, n_assets=6, seed=42)
print(m["_warning"])
save_prices_csv(m["prices"], out / "data" / "prices.csv")
save_prices_csv(m["factor_prices"], out / "data" / "factor_prices.csv")

cfg = {
    "name": "Synthetic demo portfolio",
    "holdings": {t: {"weight": float(w), "sector": s} for (t, w), s in zip(m["weights"].items(), ["Technology", "Technology", "Financials", "Healthcare", "Energy", "Industrials"])},
    "factors": {"MKT": "MKT", "SIZE": "SIZE", "VALUE": "VALUE"},
    "portfolio_value": 100_000,
    "stress": {
        # the synthetic sample has no real crises, so use windows inside it
        "scenarios": {"Synthetic window A": ["2020-03-02", "2020-04-15"], "Synthetic window B": ["2023-01-03", "2023-03-31"]},
        "shocks": {"Equity −20%": {"MKT": -0.20}, "Value +15%": {"VALUE": 0.15}},
    },
    "risk": {"n_sims": 5000},
    "backtest": {"window": 500, "refit_every": 42, "n_sims": 2000},
    "volatility": {"window": 750, "refit_every": 42},
    "optimize": {"window": 500, "rebalance_every": 126},
    "data": {"source": "csv"},
    "output": {"dir": str(out), "pdf": "--pdf" in sys.argv},
}
run(cfg, out_dir=out)
print("\nDone. These numbers come from synthetic data.")
