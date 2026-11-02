"""Stage 3, models: every risk and performance calculation.

performance  return statistics, drawdowns, concentration, sector table
factors      ETF-proxy factor model with Newey-West OLS (statsmodels)
volatility   EWMA, GARCH, GJR-GARCH (arch), out-of-sample forecast evaluation
covariance   sample, EWMA and Ledoit-Wolf (scikit-learn) covariance
risk         VaR/ES methods, Monte Carlo, Euler attribution, bootstrap intervals
stress       historical replay and hypothetical factor shocks
backtest     rolling VaR forecasts, coverage and ES tests
optimize     long-only optimizers and walk-forward evaluation
"""
