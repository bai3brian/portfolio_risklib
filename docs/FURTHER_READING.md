# Further reading and extensions

Extensions that are not implemented yet. Each item says what the method is,
why it would matter for risklib, what to read, and what a module would
produce. These are provided as reference.

## 1. Extreme Value Theory (EVT) — tail VaR/ES

**What.** Instead of modelling the whole return distribution, fit only the
tail beyond a high threshold with the Generalized Pareto Distribution
(peaks-over-threshold, POT). The shape parameter ξ tells you how heavy the
tail is; ξ > 0 means power-law tails (equities typically ξ ≈ 0.1–0.3).

**Why it matters here.** At 99% you have ~25 tail observations per decade of
data; at 99.9% you have ~2. Historical simulation cannot extrapolate; EVT
can. The standard combination is *conditional EVT*: GARCH-filter the returns
(as FHS does), then fit the GPD to the standardized residuals' tail — that
is McNeil & Frey (2000), still the reference approach.

**Read.**
- McNeil, Frey & Embrechts, *Quantitative Risk Management* (2nd ed., 2015),
  ch. 5 "Extreme Value Theory" — the chapter to read; §5.2 POT, §5.2.3
  VaR/ES from GPD.
- McNeil & Frey (2000), "Estimation of tail-related risk measures for
  heteroscedastic financial time series: an extreme value approach",
  *Journal of Empirical Finance* 7. The conditional-EVT paper; short and readable.
- Embrechts, Klüppelberg & Mikosch, *Modelling Extremal Events* (1997) —
  the theory, if you want the proofs. Not necessary.
- Practical: mean-excess plots and Hill plots for threshold choice
  (MFE ch. 5.2.4). Threshold choice is the whole art.

**What we'd build.** `extreme.py`: `fit_gpd(losses, threshold)` (MLE for
ξ, β), `evt_var_es(alpha)`, `mean_excess_plot`, plus an `"evt"` and
`"cond_evt"` method in the backtest table.

## 2. DCC-GARCH — time-varying correlation

**What.** Dynamic Conditional Correlation (Engle 2002): fit a univariate
GARCH to each asset, then model the correlation matrix of the standardized
residuals with its own GARCH-like recursion (two parameters a, b). Gives a
full covariance matrix that changes every day.

**Why it matters here.** The covariance module is static (sample, EWMA,
Ledoit-Wolf). Correlations rise in crashes — "diversification disappears
when you need it." DCC captures that and feeds directly into VaR,
Monte Carlo (`cov=` argument) and the optimizer.

**Read.**
- Engle (2002), "Dynamic Conditional Correlation: A Simple Class of
  Multivariate GARCH Models", *JBES* 20(3). The original; the two-step
  estimator is in §3.
- Engle & Sheppard (2001), "Theoretical and Empirical Properties of Dynamic
  Conditional Correlation Multivariate GARCH", NBER WP 8554 — practical
  estimation details and the correlation-targeting trick.
- Aielli (2013), "Dynamic Conditional Correlation: On Properties and
  Estimation", *JBES* — the cDCC correction; know it exists.
- Tsay, *Analysis of Financial Time Series*, ch. 10 for a textbook treatment.

**What we'd build.** `dcc.py`: `DCC.fit(returns)` reusing `volatility.GARCH`
for stage 1, `forecast_cov(h)`, `correlation_path()`; then a `"dcc"` row in
the report's covariance options.

## 3. Realized volatility & HAR (a cheap, strong benchmark)

**What.** If you have intraday data, realized variance (sum of squared
5-minute returns) is a far better volatility proxy than squared daily
returns, and the HAR-RV model (Corsi 2009) — a linear regression of RV on
its daily, weekly and monthly averages — usually beats GARCH out-of-sample.
Even with daily data, the *Garman-Klass* or *Parkinson* range estimators
from OHLC prices (which yfinance gives you) are a big improvement over
squared close-to-close returns as the *evaluation proxy*.

**Why it matters here.** The QLIKE comparison uses squared daily returns as
the realized proxy, which is noisy; an OHLC range proxy makes the model
ranking much sharper.

**Read.**
- Corsi (2009), "A Simple Approximate Long-Memory Model of Realized
  Volatility", *J. Financial Econometrics* 7(2).
- Parkinson (1980) and Garman & Klass (1980) on range estimators; any
  summary (e.g. the "volatility estimators" note by Bennett & Gil, Santander
  2012) is enough.
- Andersen, Bollerslev, Diebold & Labys (2003), "Modeling and Forecasting
  Realized Volatility", *Econometrica* — the foundational RV paper.

**What we'd build.** `download_ohlc` in `data.py`, `realized.py` with
`parkinson`, `garman_klass`, `HAR`. A small effort with a likely large
QLIKE improvement.

## 4. Regime-switching (Markov-switching / HMM)

**What.** Returns come from two (or more) regimes — calm and turbulent —
with a Markov chain switching between them (Hamilton 1989). A Gaussian HMM
on returns gives regime probabilities over time and regime-specific
vol/correlation.

**Why.** Adds a "what regime are we in today?" panel and a regime-conditional
VaR. The statistical value over GARCH for VaR is usually modest.

**Read.** Hamilton (1989), *Econometrica* 57; Ang & Timmermann (2012),
"Regime Changes and Financial Markets", *Annual Review of Financial
Economics* — the survey to read first. `hmmlearn` implements the EM step.

## 5. CoVaR / systemic risk

**What.** Adrian & Brunnermeier (2016), "CoVaR", *AER*: the VaR of the
system (or a portfolio) conditional on one asset being at its VaR,
estimated by quantile regression. ΔCoVaR measures how much one holding
adds to tail risk *conditionally* — a different lens than Euler component
VaR.

**Why.** Relevant for a concentrated portfolio, though more of a
systemic-risk research topic than a risk-desk tool. Read the paper's §2–3
and Koenker's quantile-regression intro (`statsmodels.QuantReg`).

## 6. ML for volatility / covariance

**What.** Gradient boosting or LSTMs on lagged RV, ranges and volume;
shrinkage-by-learning for covariance. Results in the literature are mixed;
HAR is a hard benchmark to beat.

**Read.** Bucci (2020), "Realized Volatility Forecasting with Neural
Networks", *J. Financial Econometrics*; Gu, Kelly & Xiu (2020), "Empirical
Asset Pricing via Machine Learning", *RFS* (for method hygiene: rolling
retraining, no look-ahead). Treat as a stretch goal; only worth it if the
HAR benchmark is already in.

## Where to find the papers
Most are on SSRN, the authors' pages or NBER. The *Quantitative Risk
Management* chapters (5 for EVT, 14 for multivariate GARCH) are the most
efficient starting point. New modules (`extreme.py`, `dcc.py`,
`realized.py`) would follow the same test discipline as the rest: recover
known parameters from simulated data.
