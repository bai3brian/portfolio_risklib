# risklib user guide

`risklib` follows the workflow of an institutional risk system: clean data,
factor model, volatility and covariance, VaR/ES, stress, backtest,
allocation, and one report. This guide covers installation, running the
report on a portfolio, the configuration options, reading the output, and
using the modules directly.

Market data comes from yfinance (or a CSV). Synthetic data exists only in
`risklib/synthetic.py`, `tests/` and `examples/demo_offline.py`; the report
pipeline never uses it.

---

## 1. Install

```bash
cd risklib
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                             # 116 tests, no network needed
```

Requirements: Python ≥ 3.10, numpy, pandas ≥ 2.2, scipy, matplotlib, PyYAML,
yfinance. Optional: a LaTeX engine for the PDF report, and `arch` for
`volatility.crosscheck_arch`. Installing the package adds a `risklib`
command; `python -m risklib` is equivalent.

## 2. Run the report on your portfolio

1. Copy `examples/portfolio.yaml` to `my_portfolio.yaml` (git-ignored).
2. Replace the `holdings` block with your positions:

   ```yaml
   holdings:
     AAPL: {shares: 10, sector: Hardware}   # valued at the last close
     JPM:  {dollars: 4000}                  # dollar amount
     VTI:  {weight: 0.3}                    # or a weight; all are normalized to sum to 1
   ```

   `sector` is optional free text for the sector table; without it the
   report looks the sector up on yfinance once and caches it in
   `data/sectors.json`. Portfolio value is computed from shares × last close
   (or dollars); set `portfolio_value` only to override it.

3. Run:

   ```bash
   risklib report my_portfolio.yaml          # add --pdf for the PDF report
   ```

   Takes 1–3 minutes depending on history length and `n_sims`. Output lands
   in `reports/` (or `output.dir`):

   | file | contents |
   |---|---|
   | `report.md` | the full report with all tables and charts |
   | `report.pdf` | health snapshot, risk summary and full detail (with `--pdf`) |
   | `metrics.json` | every headline number, machine-readable |
   | `charts/*.png` | wealth/drawdown, conditional vol, VaR backtest, component VaR, stress, walk-forward |
   | `*.csv` | each main table |
   | `data/prices.csv`, `data/factor_prices.csv` | the raw download, so you can re-run offline |

4. Re-run without downloading (same data, new settings):

   ```bash
   risklib report my_portfolio.yaml --offline
   ```

Other commands: `risklib check my_portfolio.yaml` validates the config,
`risklib fetch my_portfolio.yaml` downloads only, and
`risklib pdf reports/report.md` builds the PDF from an existing report.

### Using a CSV instead of yfinance

Any CSV with a `date` column and one numeric column per ticker works (broker
export, Bloomberg, etc.). Set:

```yaml
data:
  source: csv
  prices_csv: path/to/prices.csv          # must contain every holding
  factor_csv: path/to/factor_prices.csv   # must contain every factor proxy ticker
```

## 3. Configuration reference

Everything below is optional except `holdings`. Defaults are in
`risklib/report.py::DEFAULTS`.

### Top level
| key | default | meaning |
|---|---|---|
| `name` | "portfolio" | label in the report |
| `start` / `end` | 2015-01-01 / today | estimation window. Stress replays download further back on their own (`stress.history_start`) |
| `portfolio_value` | null (computed) | override for the value implied by the holdings; a note appears if it differs by > 5% |
| `rf_annual` | 0.0 | risk-free rate for Sharpe/Sortino and max-Sharpe |

### Holdings with short histories: `proxy`
A holding younger than the sample can carry `proxy: TICKER`:

```yaml
CEG: {shares: 5, proxy: EXC}
```

Before CEG's first real price, its price path is reconstructed so that its
daily log returns equal those of EXC, its former parent. Real data is never changed. This keeps the
panel at full length instead of cutting every holding to the youngest one
(`common: true` would otherwise do exactly that). The proxy ticker is
downloaded automatically; the report's section 1 lists each proxied holding,
the date its real data starts, and the share of the sample that is proxy.

Choosing a proxy: the former parent for a spin-off (CEG→EXC, GEV→GE); a
sector ETF for an IPO; a thematic ETF for a company without a close peer. A bad proxy makes early-history numbers
(stress replays, long-window estimates) unreliable for that name, so keep
proxied holdings small or check how much of the result they drive via the
component-VaR table.

### `panel`: data cleaning
| key | default | meaning |
|---|---|---|
| `max_gap` | 5 | forward-fill up to this many consecutive missing days; longer holes stay missing |
| `min_obs` | 252 | drop a ticker with fewer non-missing prices (the report then errors and tells you which) |
| `common` | true | truncate to dates where every ticker has a price. A recent IPO shortens everyone's history; consider removing it or lowering `start` |

### `factors`
`DEFAULT` gives MKT (SPY), SIZE (IWM−SPY), VALUE (IWD−IWF), MOM (MTUM−SPY),
SEMIS (SMH−SPY). Any mapping works: a string is a long position, a
two-element list is long-minus-short. `factors.OPTIONAL_FACTORS` lists
extras (`RATES: TLT`, `TECH: [XLK, SPY]`, `ENERGY`, `GOLD`, `USD`, `CREDIT`).
A TLT factor is *not* in the default because its daily beta to equities is
≈ 0; rate risk for a growth book is carried by VALUE and mapped in the rate
shocks. Keep it under ~8 factors or the betas get noisy. Shock names in
`stress.shocks` must match these names.

Holdings with a `proxy:` have their betas estimated on **real days only**
(when at least 250 exist); the per-asset table's `real_only` and `n_obs`
columns say which.

### `volatility`
| key | default | meaning |
|---|---|---|
| `window` | 1000 | trailing days used to fit each rolling forecast (capped at half the sample) |
| `refit_every` | 21 | days between GARCH refits in the out-of-sample comparison |

### `risk`
| key | default | meaning |
|---|---|---|
| `alphas` | [0.99, 0.975] | confidence levels. 0.975 ES is the FRTB standard |
| `horizons` | [1, 10] | days. 10-day is the Basel market-risk horizon |
| `methods` | normal, historical, fhs, mc_t_copula | rows of the VaR/ES table; also available: `t`, `hist_weighted`, `mc_normal`, `mc_t` (see `risk.ALL_METHODS`) |
| `n_sims` | 20000 | Monte Carlo / FHS paths |
| `seed` | 0 | RNG seed for reproducibility |
| `n_boot` | 300 | bootstrap resamples for the 1-day Historical / FHS confidence intervals |
| `attribution_cov` | ewma | covariance for component VaR: `ewma` (conditional) or `ledoit_wolf` (full-sample) |

### `backtest`
| key | default | meaning |
|---|---|---|
| `alpha` | 0.99 | VaR level to backtest |
| `window` | 500 | days of history each daily VaR forecast may see |
| `refit_every` | 21 | GARCH / t refit frequency |
| `methods` | normal, historical, garch_normal, fhs | also available: `t, hist_weighted, ewma, gjr_normal, garch_t` (see `backtest.ALL_METHODS`; `garch_t` is by far the slowest) |

Backtest length = sample − window. With `start: 2015` and window 500 you
get ~2,400 out-of-sample days, which is plenty for Kupiec at 99%.

### `stress`
| key | default | meaning |
|---|---|---|
| `scenarios` | built-in | `name: [start, end]` windows. Assets without data in a window are proxied through their factor betas (`proxied` column) |
| `shocks` | built-in | `name: {FACTOR: return, ...}`. Built-in rate shocks are mapped through MKT and VALUE using `stress.RATE_SHOCK_PER_100BP` (2022-calibrated: MKT −10%, VALUE +7% per +100bp) |
| `history_start` | auto | how far back prices are downloaded for the replays: `auto` (30 days before the earliest scenario), a date, or null (only from `start`). Estimation still uses data from `start` onward |

Each holding's replay return uses, in order: its own prices in the window
(`direct`); its proxy-backfilled history if the window predates its listing
(`backfilled`); its factor betas times the factor returns over the window
(`proxied`); otherwise it contributes zero (`missing`). Factors whose ETFs did
not trade in a window (MTUM, for example, starts in 2013) are left out of the
`proxied` mapping and listed in `factors_missing`.

### `optimize`
| key | default | meaning |
|---|---|---|
| `window` | 500 | estimation window per rebalance |
| `rebalance_every` | 63 | holding period (63 ≈ quarterly) |
| `w_min` / `w_max` | 0 / 1 | box constraints. `w_max: 0.25` caps any single name |
| `strategies` | min_variance, risk_parity | run for both sample and Ledoit-Wolf covariance. `max_sharpe` can be added but is not recommended: its out-of-sample result is dominated by return-estimation error |
| `expected_returns` | shrunk | `mean`, `shrunk` (James-Stein) or `zero`; only matters if `max_sharpe` is in `strategies` |

### `output` and `pdf`
| key | default | meaning |
|---|---|---|
| `output.dir` | reports | output directory |
| `output.charts` | true | write PNG charts |
| `output.pdf` | false | also compile `report.pdf` (same as `--pdf`) |
| `pdf.thresholds` | see below | limits that colour the health checklist |

The PDF is built from `report.md` alone, so it shows exactly the Markdown
numbers. Page 1 is the health snapshot: eight headline tiles and a
ten-line traffic-light checklist, each linked to its detail section. Page 2
is the risk summary (VaR/ES with confidence intervals, top contributors and
hedges, exposures, sectors, stress, backtest, allocation reference). The
remaining pages are the full report with charts. It needs `pdflatex` on the
PATH (MiKTeX on Windows, MacTeX, or TeX Live); without one, `report.tex` is
still written and can be compiled on Overleaf with the `charts/` folder.

| check | value | default amber / red |
|---|---|---|
| Risk model validity | best backtested model: coverage and ES test p-values | amber if one test fails, red if both |
| Daily VaR vs limit | 1-day 99% FHS VaR / `var_limit.limit` (5%) | ≥ 80% / ≥ 100% of the limit |
| Largest risk contributor | top holding's share of component VaR | ≥ 20% / ≥ 30% |
| Diversification | effective positions by risk | < 10 / < 5 |
| Sector concentration | top sector's share of VaR | ≥ 40% / ≥ 60% |
| Market beta | portfolio MKT exposure | outside 0.8–1.2 / outside 0.5–1.5 |
| Worst historical stress | worst replayed scenario with data | ≤ −20% / ≤ −35% |
| 1-year max drawdown | trailing 1y | ≤ −15% / ≤ −25% |
| Volatility regime | FHS VaR / historical VaR | ≥ 1.10 / ≥ 1.30 |
| Data quality | weight in holdings > 50% proxy-backfilled | ≥ 10% / ≥ 25% |

The limits are conventions, not rules; set them to your own risk appetite
under `pdf: thresholds:`. Grey means the input was missing.

## 4. Reading the report

**Portfolio.** The holdings table shows weight and VaR share side by side;
a holding whose risk share exceeds its weight share is a concentrator. The
effective number of positions (1/Herfindahl) by risk is the number of
independent bets you actually hold. **Risk profile** (vol, drawdown, skew,
kurtosis) describes the distribution's shape and is legitimate to quote;
the **trailing-window** table is what to compare with a benchmark. The
full-sample return figures are in the appendix because they apply today's
weights to a decade of history, which is hindsight.

**Factor model.** `Portfolio exposures` are your betas: MKT 1.1 means you move
110% of the market. `Variance decomposition` shows how much of your risk is
market vs idiosyncratic; a high idiosyncratic share means stock-specific risk
dominates and diversification would help. HAC t-stats above ~2 are
significant.

**Volatility.** The opening line compares today's GARCH(1,1) volatility with
its long-run level; a half-life of weeks is normal for equities. The
out-of-sample table is the honest comparison of forecasters: lower QLIKE is
better; `dm_pvalue` < 0.05 means the difference from EWMA is statistically
real.

**VaR/ES table.** Compare methods at the same alpha/horizon. Normal VaR below
historical/FHS at 99% is the fat-tail effect. FHS is the best default
estimate for "what could I lose tomorrow". `VaR_lo`/`VaR_hi` are bootstrap
95% intervals; quote VaR to the precision they support. Multi-day historical
rows are √h-scaled from 1-day (overlapping 10-day quantiles are too few to
be meaningful). `Component VaR` uses the conditional (EWMA) covariance so
it matches today's volatility; `pct_of_var` above a holding's weight means
concentrated risk.

**Stress.** Historical replay is at *today's* weights. `direct`,
`backfilled` and `proxied` tell you how much of the number is the holdings'
own prices, proxy history, or factor-implied (which omits stock-specific
moves).

**Backtest.** `breach_rate_%` should be near 1% at 99%. `kupiec_p` > 0.05:
right frequency. `christoffersen_p` > 0.05: breaches don't cluster. `cc_p`:
both. `garch_normal` and `fhs` share the GARCH volatility forecast, so the
difference between them is the tail assumption. `traffic_light_250d` is the
Basel zone on the last 250 days; `traffic_light_full` applies the same
thresholds to the whole backtest and is the one to trust. `es_Z2` > 0 means
realized losses beyond VaR exceeded ES (ES too small); `es_p` is the
one-sided bootstrap probability.

**Allocation reference.** `Walk-forward` is out-of-sample: every weight was
chosen without seeing the period it was applied to. Compare
`ledoit_wolf/risk_parity` and `ledoit_wolf/min_variance` to `current` and
`equal_weight`. `avg_turnover` is Σ|Δw| per rebalance (buys plus sells).
Max-Sharpe is available in `optimize.py` but not reported, because
its out-of-sample results are dominated by expected-return estimation error.

## 5. Using the modules in a notebook

```python
from risklib.data import load_panel, coverage_summary
from risklib.portfolio import Portfolio
from risklib import factors, volatility, covariance, risk, stress, backtest, optimize

# data (network)
prices, returns = load_panel(["AAPL", "MSFT", "VTI"], start="2015-01-01", common=False)
coverage_summary(prices)

# portfolio
port = Portfolio.from_prices(prices, {"AAPL": {"shares": 10}, "MSFT": 3000, "VTI": 5000})
port.performance(rf_annual=0.04)
ps = port.portfolio_returns("simple")

# factors
fp, fr = load_panel(factors.factor_tickers(), start="2015-01-01")
F = factors.build_factor_returns(fr)
fm = factors.FactorModel().fit(port.returns, F)
fm.summary(); fm.portfolio_exposures(port.weights); fm.risk_decomposition(port.weights)

# volatility
g = volatility.GJRGARCH().fit(port.portfolio_returns("log"))
g.summary(); g.forecast(10)
scores, fc = volatility.compare_models(port.portfolio_returns("log"), window=1000)

# covariance
S, shrink = covariance.ledoit_wolf(port.returns)

# VaR / ES
risk.parametric_normal(ps, 0.99); risk.historical(ps, 0.99, horizon=10)
risk.filtered_historical(ps, 0.99)[:2]
risk.monte_carlo(port.returns, port.weights, 0.99, dist="t_copula", cov=S)[:2]
risk.component_var_parametric(S, port.weights, 0.99)

# stress
stress.historical_replay(prices, port.weights, factor_prices=fp, factor_spec=factors.DEFAULT_FACTORS, betas=fm.betas)
stress.factor_shock(fm, port.weights, {"crash": {"MKT": -0.25, "SEMIS": -0.10}})

# backtest
table, forecasts = backtest.backtest_table(ps, methods=("historical", "fhs"), alpha=0.99, window=500)

# optimization
optimize.risk_parity(S); optimize.min_variance(S, w_max=0.3)
summary, oos, weights = optimize.walk_forward(port.returns, port.weights)
```

All models follow the same protocol: `.fit(returns)` then `.forecast(h)` /
`.conditional_variance()`. Every table is a pandas object so it can be
sliced, plotted or exported directly.

## 6. Customizing beyond the config

* **New factor**: add a line under `factors`. Nothing else changes.
* **New stress window**: add to `stress.scenarios`. If your holdings did not
  exist then, the replay uses factor betas automatically.
* **New VaR method**: add a branch to `backtest.rolling_var` and a row in
  `risk.RiskTable.run`; both are plain if/elif blocks.
* **New volatility model**: implement `fit / conditional_variance / forecast`
  (see `EWMA` for the minimal version) and add it to `compare_models`.
* **New optimizer objective**: write a function returning a weight Series and
  add a branch in `optimize.walk_forward`.
* **Shorts / leverage**: `normalize_weights` enforces long-only; relaxing it
  means changing that check, the optimizer bounds, and `component_var_*`
  (which are already sign-correct).

## 7. Common problems

| symptom | cause / fix |
|---|---|
| `yfinance returned no data` | ticker typo, delisted, or no internet. Try `python -m risklib fetch` |
| `tickers dropped for short history` | a holding is younger than `panel.min_obs` days; give it a `proxy:`, lower `min_obs`, or drop it |
| Backtest has few days | `backtest.window` too large for the sample; it is capped at half the sample |
| GARCH persistence ≈ 1 | normal for daily equities over long samples; the model is near-IGARCH. Nothing is broken |
| max-Sharpe puts 100% in one asset | expected returns are noisy. Use `w_max`, `expected_returns: shrunk`, or judge by the walk-forward |
| Charts look empty | `output.charts: false` or matplotlib missing |
| `no LaTeX engine on PATH` | install MiKTeX or TeX Live and reopen the terminal, or compile `report.tex` on Overleaf |
| `LaTeX failed:` with a log excerpt | usually a missing package on MiKTeX: allow it to install missing packages on the fly |
