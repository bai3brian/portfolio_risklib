"""Stress testing: historical replay and hypothetical factor shocks.

Both are expressed as a portfolio simple return (negative = loss) with
per-asset contributions:

historical_replay   Apply today's weights to the actual asset returns over a
                    past crisis window, using proxy history or factor betas
                    for holdings that did not trade then.
factor_shock        Hypothetical: shock the factors (e.g. MKT −20%) and map
                    through the factor model, r_p = βₚ · f.

Built-in windows are in :data:`HISTORICAL_SCENARIOS` (peak to trough for the
S&P 500); override them under ``stress.scenarios`` in the config.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# Windows run from the S&P 500 closing high to the closing low of each episode.
# The dot-com window starts at the September 2000 high rather than March 2000
# so that the size, value and semiconductor factor ETFs (listed mid-2000)
# can map holdings that did not trade yet.
HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "Dot-com bust (Sep 2000-Oct 2002)": ("2000-09-01", "2002-10-09"),
    "9/11 (Sep 2001)": ("2001-09-10", "2001-09-21"),
    "GFC peak to trough (Oct 2007-Mar 2009)": ("2007-10-09", "2009-03-09"),
    "GFC Lehman (Sep-Nov 2008)": ("2008-09-12", "2008-11-20"),
    "Flash crash / euro crisis (Apr-Jul 2010)": ("2010-04-23", "2010-07-02"),
    "US downgrade (Jul-Aug 2011)": ("2011-07-22", "2011-08-08"),
    "Taper tantrum (May-Jun 2013)": ("2013-05-21", "2013-06-24"),
    "China deval / Aug 2015": ("2015-08-17", "2015-08-25"),
    "Oil and high-yield selloff (Nov 2015-Feb 2016)": ("2015-11-03", "2016-02-11"),
    "Volmageddon (Jan-Feb 2018)": ("2018-01-26", "2018-02-08"),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Sep 2020 tech selloff": ("2020-09-02", "2020-09-23"),
    "2022 rate shock (Jan-Oct 2022)": ("2022-01-03", "2022-10-12"),
    "SVB / regional banks (Mar 2023)": ("2023-03-08", "2023-03-13"),
    "2023 rate spike (Aug-Oct 2023)": ("2023-07-31", "2023-10-27"),
    "Aug 2024 vol spike": ("2024-07-16", "2024-08-05"),
    "DeepSeek shock (Jan 2025)": ("2025-01-24", "2025-01-27"),
    "Tariff shock (Feb-Apr 2025)": ("2025-02-19", "2025-04-08"),
}

# Hypothetical factor shocks (simple returns of the factor proxies).
#
# Rate shocks are mapped through equities the way 2022 transmitted them, as a
# market fall plus a value-over-growth rotation, rather than through a
# bond-price factor whose equity beta is near zero. Calibration over
# 2022-01-03 to 2022-10-12 (10y yield about +240bp): SPY about −24%, IWD−IWF
# about +17pp, i.e. per +100bp MKT −10% and VALUE +7%.
RATE_SHOCK_PER_100BP: dict[str, float] = {"MKT": -0.10, "VALUE": 0.07}

HYPOTHETICAL_SHOCKS: dict[str, dict[str, float]] = {
    "Equity −10%": {"MKT": -0.10},
    "Equity −20%": {"MKT": -0.20},
    "Equity −30% (1987/2008-type)": {"MKT": -0.30},
    "Rates +100bp (2022-calibrated: MKT −10%, VALUE +7%)": {k: v for k, v in RATE_SHOCK_PER_100BP.items()},
    "Rates +200bp (2022-calibrated: MKT −20%, VALUE +14%)": {k: 2 * v for k, v in RATE_SHOCK_PER_100BP.items()},
    "Growth crash: value +15%, market −10%": {"VALUE": 0.15, "MKT": -0.10},
    "Momentum crash (2009-type: MOM −25%)": {"MOM": -0.25},
    "Semiconductor bust: SEMIS −30%, market −10%": {"SEMIS": -0.30, "MKT": -0.10},
    "Small-cap liquidation: SIZE −15%, market −10%": {"SIZE": -0.15, "MKT": -0.10},
}


def _window_return(prices: pd.DataFrame, start: str, end: str) -> pd.Series | None:
    """Simple return of each column from the last price ≤ start to the last ≤ end."""
    p = prices.loc[:end]
    if p.empty:
        return None
    p0 = prices.loc[:start]
    if p0.empty:
        return None
    return p.iloc[-1] / p0.iloc[-1] - 1.0


def historical_replay(
    asset_prices: pd.DataFrame,
    weights: pd.Series,
    scenarios: Mapping[str, tuple[str, str]] = HISTORICAL_SCENARIOS,
    factor_prices: pd.DataFrame | None = None,
    factor_spec: Mapping | None = None,
    betas: pd.DataFrame | None = None,
    real_from: Mapping[str, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay historical windows through today's weights.

    Each holding's window return comes from, in order of preference:

    * ``direct``: its own prices at both ends of the window;
    * ``backfilled``: its proxy-backfilled prices, when the window starts
      before ``real_from[ticker]`` (the first real price);
    * ``proxied``: ``β_i · f`` from its factor betas and the factor returns
      over the window (idiosyncratic part zero);
    * ``missing``: none of the above (contributes zero).

    Parameters
    ----------
    asset_prices : pd.DataFrame
        Long-history asset PRICES, proxy-backfilled where configured.
    factor_prices, factor_spec, betas :
        Factor-proxy PRICES, the factor definitions and per-asset betas used
        for the ``proxied`` route. Factors whose ETFs have no prices over the
        window are left out of the mapping and listed in ``factors_missing``.
    real_from : mapping, optional
        First real price date per proxy-backfilled holding.

    Returns
    -------
    (summary, contributions)
        ``summary``: per scenario, the portfolio return and how many holdings
        took each route. ``contributions``: scenario x asset weighted returns.
    """
    from .factors import build_factor_returns

    real_from = dict(real_from or {})
    w = weights.reindex(asset_prices.columns).fillna(0.0)
    rows, contribs = [], {}
    for name, (start, end) in scenarios.items():
        asset_r = _window_return(asset_prices, start, end)
        direct = backfilled = proxied = missing = 0
        contrib = pd.Series(0.0, index=w.index)
        fr, f_missing = None, []
        if factor_prices is not None and factor_spec is not None and betas is not None:
            fr_prox = _window_return(factor_prices, start, end)
            if fr_prox is not None:
                avail = {}
                for k, spec in factor_spec.items():
                    legs = [spec] if isinstance(spec, str) else list(spec)
                    if all(leg in fr_prox.index and pd.notna(fr_prox[leg]) and bool(factor_prices[leg].loc[:start].notna().any()) for leg in legs):
                        avail[k] = spec
                f_missing = [k for k in factor_spec if k not in avail]
                if avail:
                    fr = build_factor_returns(fr_prox.to_frame().T, avail).iloc[0]
            else:
                f_missing = list(factor_spec)
        for t in w.index:
            r = asset_r[t] if asset_r is not None else np.nan
            has_start = bool(asset_prices[t].loc[:start].notna().any())
            if asset_r is not None and pd.notna(r) and has_start:
                contrib[t] = w[t] * r
                if t in real_from and pd.Timestamp(start) < pd.Timestamp(real_from[t]):
                    backfilled += 1
                else:
                    direct += 1
            elif fr is not None and t in betas.index:
                contrib[t] = w[t] * float(betas.loc[t].reindex(fr.index).fillna(0.0) @ fr)
                proxied += 1
            else:
                missing += 1
        contribs[name] = contrib
        rows.append({"scenario": name, "start": start, "end": end, "portfolio_return": float(contrib.sum()),
                     "direct": direct, "backfilled": backfilled, "proxied": proxied, "missing": missing,
                     "factors_missing": ", ".join(f_missing) if proxied else ""})
    summary = pd.DataFrame(rows).set_index("scenario")
    contributions = pd.DataFrame(contribs).T
    contributions.index.name, contributions.columns.name = "scenario", "ticker"
    return summary, contributions


def factor_shock(model, weights: pd.Series, shocks: Mapping[str, dict[str, float]] = HYPOTHETICAL_SHOCKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map hypothetical factor shocks through a fitted FactorModel.

    r_p = Σ_k βₚ,k · f_k  (idiosyncratic returns assumed zero).

    Returns
    -------
    (summary, contributions)
        ``summary``: per scenario portfolio return and the shock applied.
        ``contributions``: scenario x asset.
    """
    w = weights.reindex(model.assets).fillna(0.0)
    rows, contribs = [], {}
    for name, shock in shocks.items():
        f = pd.Series(0.0, index=model.factors)
        unknown = [k for k in shock if k not in f.index]
        for k, v in shock.items():
            if k in f.index:
                f[k] = v
        asset_r = model.betas.values @ f.values
        contrib = pd.Series(w.values * asset_r, index=model.assets)
        contribs[name] = contrib
        rows.append({"scenario": name, "portfolio_return": float(contrib.sum()), "shock": ", ".join(f"{k} {v:+.0%}" for k, v in shock.items()), "ignored_factors": ", ".join(unknown)})
    summary = pd.DataFrame(rows).set_index("scenario")
    contributions = pd.DataFrame(contribs).T
    contributions.index.name, contributions.columns.name = "scenario", "ticker"
    return summary, contributions
