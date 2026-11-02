# Risk Library for Long-Only US Equity Portfolios
### Developed by Brian Bai, btb2136@columbia.edu

![tests](https://github.com/bai3brian/portfolio_risklib/actions/workflows/tests.yml/badge.svg)

A Python risk engine for long-only US equity portfolios, modelled on the
workflow of an institutional risk system. It downloads prices from Yahoo
Finance, then runs the following steps in order: a factor model,
volatility and covariance estimation, VaR and Expected Shortfall, stress
tests, out-of-sample backtests and an allocation reference. The results come
out as one Markdown report, with an optional PDF whose first page is a
portfolio-health snapshot.

Estimation runs on the standard scientific Python stack: statsmodels for the
factor regressions and Newey-West errors, arch for the GARCH family,
scikit-learn for Ledoit-Wolf shrinkage, pandas for EWMA, and numpy/scipy for
the Monte Carlo engines, VaR methods and backtest statistics. The test suite
checks each model against data simulated from known parameters.

```bash
pip install -e ".[dev]"
pytest                                          # 115 tests, no network needed
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
| Factor model | ETF long-short factors (market, size, value, momentum, semiconductors), OLS with Newey-West HAC errors (statsmodels), variance decomposition |
| Volatility | EWMA (RiskMetrics, pandas), GARCH(1,1), GJR-GARCH and Student-t GARCH by maximum likelihood (arch); rolling out-of-sample comparison with QLIKE and Diebold-Mariano |
| Covariance | sample, EWMA (pandas), Ledoit-Wolf shrinkage (scikit-learn); PSD repair |
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
  __main__.py          command-line interface (risklib report | fetch | check | pdf)
  config.py            config defaults, YAML loading, stress-history start date
  pipeline.py          runs the four stages below end to end
  data/                1. where prices come from
    sources.py         Yahoo Finance download, CSV input/output, sector lookup
    synthetic.py       simulated markets for the tests and the offline demo
  preprocessing/       2. raw prices to a clean panel and weights
    panel.py           validation, gap filling, proxy backfill, alignment, returns
    portfolio.py       holdings to weights, portfolio returns, Portfolio object
  models/              3. every risk and performance calculation
    performance.py     return statistics, drawdowns, concentration, sector table
    factors.py         ETF-proxy factor model with Newey-West OLS
    volatility.py      EWMA, GARCH, GJR-GARCH, forecast evaluation
    covariance.py      sample, EWMA and Ledoit-Wolf covariance
    risk.py            VaR/ES methods, Monte Carlo, attribution, bootstrap intervals
    stress.py          historical replay and factor shocks
    backtest.py        rolling VaR forecasts and coverage / ES tests
    optimize.py        optimizers and walk-forward evaluation
  output/              4. results to files
    charts.py          report figures
    markdown.py        report.md
    metrics.py         metrics.json
    pdf.py             report.md to a LaTeX PDF with a health snapshot
tests/                 pytest suite (synthetic data only)
examples/              example config, offline demo, sample output
docs/                  user guide, model catalogue, further reading
```

Dependencies run one way: `data` → `preprocessing` → `models` → `output`,
with `pipeline.py` calling each stage in turn.

## Dependencies

All are installed by `pip install -e ".[dev]"` (see `pyproject.toml`).

| Package | Used in | What it does there |
|---|---|---|
| numpy | every module | Array math and linear algebra (Cholesky and eigendecompositions for the Monte Carlo engines), random numbers for Monte Carlo, FHS and bootstrap intervals, empirical quantiles for historical VaR/ES |
| pandas | every module | Price and return frames, date alignment, rolling windows, CSV input/output; `ewm` computes the EWMA volatility (`models/volatility.py`) and EWMA covariance (`models/covariance.py`) |
| scipy | `models/`: `risk.py`, `backtest.py`, `volatility.py`, `optimize.py` | `scipy.stats`: normal and Student-t quantiles, densities and fits for parametric VaR/ES, the copula simulations and the t-based backtest models; χ² and binomial distributions for the Kupiec, Christoffersen and Basel traffic-light tests; the normal p-value of the Diebold-Mariano test. `scipy.optimize.minimize`: SLSQP for the long-only optimizers, L-BFGS-B for risk parity |
| statsmodels | `models/factors.py` | `OLS` with `cov_type="HAC"`: factor betas, R² and Newey-West standard errors |
| arch | `models/volatility.py` (and through it FHS in `risk.py` and the GARCH backtests in `backtest.py`) | `arch_model`: maximum-likelihood fits of GARCH(1,1), GJR-GARCH and Student-t GARCH with robust standard errors; `fix` to roll fitted parameters forward on new data; analytic variance forecasts |
| scikit-learn | `models/covariance.py` | `LedoitWolf`: shrinkage covariance used by the Monte Carlo VaR and the walk-forward optimizers |
| matplotlib | `output/charts.py` | The report charts: conditional volatility, component VaR, stress scenarios, VaR backtest, walk-forward |
| PyYAML | `config.py` | Reads the portfolio config (`my_portfolio.yaml`) |
| yfinance | `data/sources.py` | Downloads adjusted daily prices from Yahoo Finance; looks up missing sector labels |
| pytest, ruff | `tests/` (development only) | Test suite and linting |

The PDF report (`output/pdf.py`) also needs a LaTeX engine, `pdflatex` or `latexmk` from MiKTeX, MacTeX or TeX Live. It is a system program, not a Python package; without it the report is still written as Markdown and `report.tex`.

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

Synthetic data is confined to `risklib/data/synthetic.py`, the tests and the
offline demo; the report pipeline never imports it. `my_portfolio.yaml` and
`reports/` are git-ignored so personal holdings stay out of the repository.
The output is a risk description, not investment advice.
