# Model catalogue

Every method in risklib, grouped by module in the order the report runs
them. Each entry gives the method and its source, what it does, and its
**role** in the pipeline:

* **core**: produces a headline number in the report
* **benchmark**: exists so a core method has something to be compared against
* **diagnostic**: tells you whether a core number can be trusted
* **alternative**: a second route to a number a core method already gives
* **optional**: off by default; switch on through `risk.methods` or `backtest.methods`
* **library only**: implemented and tested but not shown in the report

## Preprocessing (`preprocessing/panel.py`)

| Method | What it does | Role |
|---|---|---|
| Log returns ln(P_t/P_{t−1}) | Time-additive returns used to fit every model; portfolio aggregation converts to simple returns first. | core |
| Capped forward-fill | Carries the last price forward for at most `max_gap` days; never fills backwards, so there is no look-ahead. | core |
| Common-window alignment | Keeps dates on which every asset has a price, giving the rectangular panel the regressions and covariances need. | core |
| Proxy backfill | Rebuilds a young stock's pre-listing path from a related instrument's returns (former parent, peer or sector ETF). | core |

## Portfolio and performance (`preprocessing/portfolio.py`, `models/performance.py`)

| Method | What it does | Role |
|---|---|---|
| Daily-rebalanced vs buy-and-hold returns | Constant-weight returns feed every risk model, since risk describes today's weights; buy-and-hold matches a brokerage statement. | core |
| CAGR, annualized volatility (√252) | Baseline description of growth and dispersion. | core |
| Sharpe ratio (Sharpe 1966) | Mean excess return over volatility. | core |
| Sortino ratio (Sortino & van der Meer 1991) | Excess return over downside deviation. | diagnostic |
| Maximum drawdown, Calmar ratio | Largest peak-to-trough loss; CAGR / abs(MDD). | core |
| Skewness, excess kurtosis | Third and fourth standardized moments; fat tails motivate the non-normal models below. | diagnostic |
| Trailing-window performance | The same statistics over the last 1, 3 and 5 years. | core |
| Herfindahl index / effective number of positions (Hirschman 1945; Meucci 2009) | 1/Σs_i² on weight shares and on VaR shares: how many independent bets the book holds. | core |
| Sector aggregation | Weight and VaR share by sector. | core |

## Factor model (`models/factors.py`)

| Method | What it does | Role |
|---|---|---|
| Linear multi-factor model (Sharpe 1964; Fama & French 1993; Carhart 1997; BARRA-style sector factor) | r_i = α_i + Σ_k β_ik f_k + ε_i with ETF factors: SPY, IWM−SPY, IWD−IWF, MTUM−SPY, SMH−SPY. Proxied holdings are estimated on real days only. | core |
| Ordinary least squares | Closed-form (XᵀX)⁻¹Xᵀy beta estimates (statsmodels `OLS`). | core |
| Newey-West HAC standard errors (Newey & West 1987, 1994) | Coefficient covariance robust to heteroskedasticity and autocorrelation; Bartlett kernel with L = ⌊4(n/100)^{2/9}⌋ (statsmodels `cov_type="HAC"`, no small-sample correction). | diagnostic |
| Variance decomposition | Var(r_p) = βₚᵀFβₚ + Σ w_i²σ²_ε,i, split by factor with the Euler allocation β_p,k(Fβ_p)_k. | core |

## Volatility (`models/volatility.py`)

| Method | What it does | Role |
|---|---|---|
| EWMA / RiskMetrics (J.P. Morgan 1996) | σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1} with λ = 0.94; the industry baseline. | benchmark |
| GARCH(1,1) (Engle 1982; Bollerslev 1986) | σ²_t = ω + αε²_{t−1} + βσ²_{t−1}: mean-reverting volatility clustering; supplies the σ forecast for FHS and the GARCH backtests. | core |
| GJR-GARCH (Glosten, Jagannathan & Runkle 1993) | Adds γ·1[ε<0]ε², the leverage effect; one of the forecasters in the out-of-sample comparison. | benchmark |
| Student-t GARCH (Bollerslev 1987) | Same recursion with standardized-t innovations; separates fat tails from clustering (backtest model `garch_t`). | optional |
| Maximum likelihood | Parameters chosen to maximize the normal or standardized-t log-likelihood, fitted with the `arch` package on percent returns and converted back to return units. | core |
| Robust (QMLE) standard errors (Bollerslev & Wooldridge 1992) | arch's default sandwich covariance, valid when the innovation distribution is misspecified. | diagnostic |
| QLIKE and MSE loss (Patton 2011) | Score variance forecasts against squared returns; QLIKE is robust to the noisy proxy. | diagnostic |
| Diebold-Mariano test (1995) | HAC t-test on the loss differential between two forecasters. | diagnostic |
| Rolling out-of-sample refit | Refit on a trailing window every 21 days and forecast one day ahead. | diagnostic |

## Covariance (`models/covariance.py`)

| Method | What it does | Role |
|---|---|---|
| Sample covariance | Unbiased estimator; baseline. | core |
| EWMA covariance | RiskMetrics recursion on outer products (pandas `ewm`); used for conditional component VaR. | core |
| Ledoit-Wolf shrinkage to scaled identity (2004) | Σ* = (1−δ)S + δμI with the analytic optimal δ (scikit-learn `LedoitWolf`); used for Monte Carlo and optimization. | core |
| PSD repair | Clips negative eigenvalues before Cholesky. | diagnostic |

## VaR, ES and simulation (`models/risk.py`)

| Method | What it does | Role |
|---|---|---|
| Value-at-Risk (RiskMetrics 1994; Jorion 1996) | Loss quantile at confidence α. | core |
| Expected Shortfall (Artzner et al. 1999; Acerbi & Tasche 2002) | Mean loss beyond VaR; coherent, and the Basel FRTB measure at 97.5%. | core |
| Parametric normal VaR/ES | z_ασ and σφ(z)/(1−α). | benchmark |
| Parametric Student-t VaR/ES | ν and scale by MLE; closed-form ES. | optional |
| Historical simulation | Empirical quantile of past returns. | core |
| Age-weighted historical simulation (Boudoukh, Richardson & Whitelaw 1998) | Scenario weights decay with age (λ = 0.99). | optional |
| Filtered historical simulation (Barone-Adesi, Giannopoulos & Vosper 1999) | Bootstraps GARCH-standardized residuals and rescales them by the current GARCH σ: empirical tails at today's volatility. | core |
| Bootstrap confidence intervals (Efron 1979) | Resample days (historical) or standardized residuals (FHS) to put a 95% interval on VaR and ES. | diagnostic |
| √h scaling | Multi-day parametric and historical figures scaled from 1-day; FHS and Monte Carlo simulate the horizon directly. | core |
| Cholesky factorization | Σ = LLᵀ; correlated draws are Lz. | core |
| Multivariate normal Monte Carlo | Should reproduce parametric VaR; a consistency check. | optional |
| Multivariate Student-t Monte Carlo | Covariance-matched joint fat tails. | optional |
| Gaussian copula with empirical marginals (Sklar 1959; Li 2000) | Normal dependence, historical marginal distributions kept exactly. | library only |
| t-copula with empirical marginals (Demarta & McNeil 2005) | Adds tail dependence: joint crashes are more likely. | core |
| Euler component VaR (Litterman 1996; Tasche 2000) | VaR = Σ w_i ∂VaR/∂w_i; which holding drives risk, on the conditional (EWMA) covariance. | core |

## Stress testing (`models/stress.py`)

| Method | What it does | Role |
|---|---|---|
| Historical scenario replay | Today's weights applied to real returns over past crisis windows; assets without data are mapped through their factor betas. | core |
| Factor-shock mapping | r_p = βₚ·f for hypothetical factor moves. Rate shocks map through MKT and VALUE using a 2022-calibrated transmission (−10% / +7% per +100bp). | core |

## Backtesting (`models/backtest.py`)

| Method | What it does | Role |
|---|---|---|
| Rolling out-of-sample VaR | Each day's VaR uses only the prior window (default 500 days); four models by default (normal, historical, GARCH-normal, FHS); t, age-weighted, EWMA, GJR-normal and GARCH-t are optional. | core |
| Kupiec proportion-of-failures test (1995) | Likelihood-ratio test that the breach rate equals 1−α. | core |
| Christoffersen independence test (1998) | Markov-chain test that breaches do not cluster. | core |
| Conditional coverage (Christoffersen 1998) | LR_uc + LR_ind ~ χ²(2); the pass/fail criterion for model selection. | core |
| Basel traffic light (BCBS 1996) | Green ≤ 4, yellow 5–9, red ≥ 10 breaches per 250 days at 99%, plus the same binomial zones over the full backtest. | diagnostic |
| Acerbi-Székely ES test Z₂ (2014) | Tests whether losses beyond VaR match the forecast ES; one-sided bootstrap p-value. | core |

## Allocation (`models/optimize.py`)

| Method | What it does | Role |
|---|---|---|
| Mean-variance framework (Markowitz 1952) | Trade-off between wᵀμ and wᵀΣw. | core |
| Minimum-variance portfolio | argmin wᵀΣw, long-only; needs no expected returns. | core |
| Risk parity / equal risk contribution (Maillard, Roncalli & Teïletche 2010; Spinu 2013) | Each asset contributes equal risk; solved in Spinu's convex form. | core |
| Maximum-Sharpe (tangency) portfolio (Tobin 1958) | argmax (wᵀμ − r_f)/√(wᵀΣw). | library only |
| Target-volatility portfolio | max wᵀμ subject to √(wᵀΣw) ≤ σ*. | library only |
| Efficient frontier | Minimum variance for each target return. | library only |
| James-Stein shrinkage of expected returns (Jorion 1986) | Pulls sample means toward the grand mean. | diagnostic |
| SLSQP | scipy's constrained optimizer for the long-only problems. | core |
| Walk-forward evaluation (cf. Bailey & López de Prado 2014) | Re-estimate on 500 days, hold 63, repeat; only held-out performance is reported. | core |
