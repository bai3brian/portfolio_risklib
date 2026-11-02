# models

Stage 3 of the pipeline: every risk and performance calculation.
[`docs/MODEL_CATALOGUE.md`](../../docs/MODEL_CATALOGUE.md) lists each method with its reference.

| File | Contents | Package |
|---|---|---|
| `performance.py` | Return statistics, drawdowns, trailing windows, concentration (effective positions), sector table | pandas |
| `factors.py` | Five-factor ETF-proxy model, betas with Newey-West standard errors, variance decomposition | statsmodels |
| `volatility.py` | EWMA, GARCH(1,1), GJR-GARCH, Student-t GARCH; out-of-sample QLIKE and Diebold-Mariano comparison | arch |
| `covariance.py` | Sample, EWMA and Ledoit-Wolf covariance | scikit-learn |
| `risk.py` | VaR/ES (parametric, historical, filtered historical, Monte Carlo copulas), bootstrap intervals, Euler component VaR | scipy |
| `stress.py` | Historical replay of 19 crises (2000–2025) and hypothetical factor shocks | |
| `backtest.py` | Rolling out-of-sample VaR; Kupiec, Christoffersen, Basel traffic light, Acerbi-Székely ES test | scipy |
| `optimize.py` | Minimum variance, risk parity, max-Sharpe, target volatility; walk-forward evaluation | scipy |

Previous stage: [`preprocessing/`](../preprocessing) · Next stage: [`output/`](../output)
