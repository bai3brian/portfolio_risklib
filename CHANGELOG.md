# Changelog

## 0.6.0
Decluttered report: seven sections plus the appendix (was eight), 8-page PDF
(was 10), roughly a third of the run time.
- Removed from the report and the code: covariance-estimator comparison
  (`compare_estimators`, constant-correlation Ledoit-Wolf, factor-implied
  covariance, condition number), component ES, risk-reducing trades, reverse
  stress test.
- Removed from the report only: GARCH and GJR-GARCH parameter tables, the
  Student-t ν line, the wealth/drawdown chart, in-sample reference weights and
  volatility-based risk contributions (`optimize.risk_contributions` stays in
  the library).
- Volatility section: a one-line read of current vs long-run GARCH volatility,
  the conditional-volatility chart (GARCH only, with its long-run level) and
  the out-of-sample forecast comparison.
- VaR/ES table: four methods (parametric normal, historical, FHS, MC t-copula)
  at 99% and 97.5%, 1 and 10 days; other methods via `risk.methods`.
- Backtest: four models by default (normal, historical, GARCH-normal, FHS);
  the other five via `backtest.methods`.
- Per-asset betas table drops the alpha columns.
- The PDF parser finds sections by name, so section numbers can change.

## 0.5.0
- Nineteen built-in historical scenarios from the dot-com bust to the 2025
  tariff shock (added: dot-com, 9/11, full GFC, 2010, taper tantrum,
  2015-16 oil/high-yield, Volmageddon, Sep 2020 tech, SVB, 2023 rates,
  DeepSeek, 2025 tariffs). The PDF summary lists the worst eight.
- Historical stress replays cover every scenario regardless of `start`:
  prices are downloaded back to `stress.history_start` (default: 30 days
  before the earliest scenario) while estimation still uses data from
  `start`. Each holding is replayed on its own prices, its proxy-backfilled
  history, or its factor betas; the table reports `direct`, `backfilled`,
  `proxied`, `missing` and `factors_missing`.
- PDF report (`risklib/pdfreport.py`): a one-page health snapshot with a
  ten-item traffic-light checklist and configurable thresholds, a risk summary
  page, and the full report with charts. Available as `risklib pdf`,
  `risklib report --pdf` or `output.pdf: true`.
- The Monte Carlo engines moved into `risk.py`; `montecarlo.py` is gone
  (`risk.simulate`, `risk.simulate_portfolio`, `risk.simulate_paths`).
- `risklib` console command; complete dependency list in `pyproject.toml`.
- Repository tidy-up: docs moved to `docs/`, the portfolio tests merged into
  `tests/test_data.py`, a generic example config, a regenerated synthetic
  sample report, and a CI workflow.

## 0.4.1
- GARCH-family MLE optimizes scaled parameters (ω/σ², α, β, γ, ν) with SLSQP
  under a stationarity constraint from a grid of starting points, followed by
  a Nelder-Mead polish. The previous optimizer could stall near its starting
  values, which biased GJR-GARCH estimates of γ. The variance recursion uses
  `scipy.signal.lfilter`.

## 0.4.0
- Factor betas for proxy-backfilled holdings are estimated on real days only.
- Backtest models renamed `garch_normal` and `gjr_normal`, with a new
  `garch_t` row; Basel zones reported over the last 250 days and over the
  full backtest.
- Default factors: MKT, SIZE, VALUE, MOM, SEMIS. Rate shocks are mapped
  through equities with a 2022-calibrated transmission rather than a TLT
  factor.
- Multi-day historical VaR is √h-scaled from 1-day; bootstrap confidence
  intervals added for historical and FHS VaR/ES.
- Component VaR on the EWMA covariance; risk-reducing trades table.
- Portfolio value from holdings, concentration metrics, sector table,
  trailing-window performance; full-sample return statistics moved to an
  appendix.
- Max-Sharpe, target-vol and the efficient frontier kept in
  `risklib.optimize` but removed from the report.
