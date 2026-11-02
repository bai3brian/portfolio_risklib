"""risklib: a quantitative risk library for long-only equity portfolios.

Data flows through four stages, one subpackage each:

    data/           where prices come from: Yahoo Finance, CSV, or synthetic
    preprocessing/  cleaning, alignment, returns, proxy backfill, holdings to weights
    models/         performance, factor model, volatility, covariance, VaR/ES,
                    stress tests, backtests, allocation
    output/         charts, report.md, metrics.json and the PDF

``config.py`` holds the config defaults and ``pipeline.py`` runs the stages
end to end (``risklib report``).
"""

__version__ = "0.8.0"

TRADING_DAYS = 252
