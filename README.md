# Risk Library for Long-Only US Equity Portfolios
### Developed by Brian Bai, btb2136@columbia.edu

![tests](https://github.com/OWNER/risklib/actions/workflows/tests.yml/badge.svg)

A Python risk engine for long-only US equity portfolios, modelled on the
workflow of an institutional risk system. It downloads prices from Yahoo
Finance, then runs the following steps in order: a factor model,
volatility and covariance estimation, VaR and Expected Shortfall, stress
tests, out-of-sample backtests and an allocation reference. The results come
out as one Markdown report, with an optional PDF whose first page is a
portfolio-health snapshot.

Every model is implemented from first principles in numpy/scipy, including
the GARCH likelihoods, Ledoit-Wolf shrinkage, the copula simulators and the
backtest statistics. The test suite checks each one against data simulated
from known parameters.

```bash
pip install -e ".[dev]"
pytest                                          # 116 tests, no network needed
cp examples/portfolio.yaml my_portfolio.yaml    # edit the holdings
risklib report my_portfolio.yaml --pdf          # writes reports/report.md and report.pdf
```

To try it without network access, run `python examples/demo_offline.py --pdf`.
It runs the full pipeline on a synthetic market. A sample of that output is
in [`examples/sample_report/`](examples/sample_report/)
([report.md](examples/sample_report/report.md),
[report.pdf](examples/sample_report/report.pdf)).

## What it computes

| Area | Methods |
|---|---|
| Data | adjusted prices from yfinance or CSV, capped forward-fill, common-window alignment, proxy backfill for recent IPOs and spin-offs |
| Portfolio | daily-rebalanced and buy-and-hold returns, CAGR, volatility, Sharpe, Sortino, drawdown, Calmar, trailing windows, Herfindahl / effective number of positions by weight and by risk, sector aggregation |
| Factor model | ETF long-short factors (market, size, value, momentum, semiconductors), OLS with Newey-West HAC errors, variance decomposition |
| Volatility | EWMA (RiskMetrics), GARCH(1,1), GJR-GARCH and Student-t GARCH by maximum likelihood; rolling out-of-sample comparison with QLIKE and Diebold-Mariano |
| Covariance | sample, EWMA, Ledoit-Wolf shrinkage; PSD repair |
| VaR / ES | parametric normal and Student-t, historical and age-weighted historical, filtered historical simulation, Monte Carlo (normal, Student-t, Gaussian and t copulas with empirical marginals), bootstrap confidence intervals |
| Attribution | Euler component VaR, marginal and standalone VaR, diversification benefit |
| Stress | historical crisis replay at today's weights, hypothetical factor shocks (rate shocks calibrated to 2022); 19 crises from 2000 to 2025 |
| Backtesting | rolling out-of-sample VaR (four models by default, nine available); Kupiec, Christoffersen and conditional-coverage tests, Basel traffic light, Acerbi-Székely ES test |
| Allocation | minimum variance, risk parity, max-Sharpe and target volatility (library only), James-Stein expected returns, walk-forward evaluation |

[`docs/MODEL_CATALOGUE.md`](docs/MODEL_CATALOGUE.md) lists each method with
its reference and its role in the report.

## Layout

```
risklib/
  data.py         download, cleaning, alignment, returns, proxy backfill
  portfolio.py    Portfolio object, holdings, concentration, performance statistics
  factors.py      ETF-proxy factor model with Newey-West OLS
  volatility.py   EWMA, GARCH, GJR-GARCH, forecast evaluation
  covariance.py   covariance estimators and diagnostics
  risk.py         VaR/ES methods, Monte Carlo simulations, attribution, bootstrap intervals
  stress.py       historical replay and factor shocks
  backtest.py     rolling VaR forecasts and coverage / ES tests
  optimize.py     optimizers and walk-forward evaluation
  report.py       end-to-end pipeline, charts, report.md and metrics.json
  pdfreport.py    report.md to a LaTeX PDF with a health snapshot
  synthetic.py    simulated data for tests and the offline demo
  __main__.py     command-line interface
tests/            pytest suite (synthetic data only)
examples/         example config, offline demo, sample output
docs/             user guide, model catalogue, further reading
```

## Command line

```bash
risklib report my_portfolio.yaml [--out DIR] [--offline] [--pdf]
risklib fetch  my_portfolio.yaml            # download prices only
risklib check  my_portfolio.yaml            # validate the config
risklib pdf    reports/report.md            # build the PDF from an existing report
```

`--offline` reuses the prices saved by an earlier run. The PDF needs a LaTeX
engine (MiKTeX, MacTeX or TeX Live). Without one, `report.tex` is still
written and can be compiled on Overleaf.

## Documentation

- [User guide](docs/USER_GUIDE.md): installation, configuration reference,
  how to read the report, and how to use the modules directly.
- [Model catalogue](docs/MODEL_CATALOGUE.md): every method, its source and
  why it is in the pipeline.
- [Further reading](docs/FURTHER_READING.md): extensions not yet built (EVT,
  DCC-GARCH, realized volatility) and what to read for each.
- [Changelog](CHANGELOG.md)

## Notes

Synthetic data is confined to `risklib/synthetic.py`, the tests and the
offline demo; the report pipeline never imports it. `my_portfolio.yaml` and
`reports/` are git-ignored so personal holdings stay out of the repository.
The output is a risk description, not investment advice.
