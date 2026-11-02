"""Portfolio config: YAML loading, defaults and the stress-history start date.

Every key except ``holdings`` is optional; :data:`DEFAULTS` lists them all and
docs/USER_GUIDE.md documents them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .models import backtest as bt
from .models import factors as fm
from .models import risk, stress

# ---------------------------------------------------------------------------
# defaults
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


def load_config(path: str | Path) -> dict:
    """Read a YAML portfolio config file."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# stress history
# ---------------------------------------------------------------------------
def scenarios(cfg: dict) -> dict[str, tuple[str, str]]:
    """Historical stress windows: the config's ``stress.scenarios`` or the built-in set."""
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
        first = min(pd.Timestamp(s) for s, _ in scenarios(cfg).values())
        hs = first - pd.Timedelta(days=30)
    return str(min(pd.Timestamp(hs), pd.Timestamp(cfg["start"])).date())
