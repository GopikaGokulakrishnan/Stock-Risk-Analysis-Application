# Academic Capstone Report Structure
## TITLE: Production-Grade Stock Risk Analytics and Valuation Suite using Machine Learning and Time-Series Modeling

**Degree**: Master of Computer Applications (MCA) / Financial Technology Capstone Project
**Academic Year**: 2026

---

## 📑 Table of Contents (Target length: 8 - 12 pages)

1. **Chapter 1: Introduction & Executive Overview**
   * 1.1 Problem Statement
   * 1.2 System Objectives
   * 1.3 FinTech Platform Context (Bloomberg Emulator)
2. **Chapter 2: Financial Engineering Methodology & Literature Review**
   * 2.1 Time Series Models (ARIMA)
   * 2.2 Volatility Models (GARCH)
   * 2.3 Intrinsic Valuation (Gordon Growth DCF)
   * 2.4 Stochastic Process Simulations (Geometric Brownian Motion)
   * 2.5 Market Risk Frameworks (Value at Risk & Expected Shortfall)
   * 2.6 Credit Risk Classifier Models (Logistic Sigmoid Classifiers)
   * 2.7 Portfolio Theory (Markowitz Modern Portfolio Theory)
3. **Chapter 3: System Architecture & Data Pipeline**
   * 3.1 Software Architecture & Module Decoupling
   * 3.2 yfinance Data Acquisition, Caching and Ingestion Engine
   * 3.3 Data Processing: Log returns vs Simple returns
4. **Chapter 4: Implementation & Visualizations Design**
   * 4.1 Frontend UI Interface (Streamlit layouts)
   * 4.2 Interactive Charts Engine (Plotly waterfall, frontiers, paths, heatmaps)
5. **Chapter 5: Empirical Results & Analysis**
   * 5.1 Case Study: NSE Blue-Chip Stock Analysis
   * 5.2 Volatility Regime Shift Detections
   * 5.3 Kupiec Backtest Results
   * 5.4 Credit Risk AUC-ROC Performance
6. **Chapter 6: System Limitations & Constraints**
   * 6.1 Linear Assumptions in ARIMA
   * 6.2 Normal Distribution Assumption in Parametric VaR
   * 6.3 Input Sensitivity in DCF Models
7. **Chapter 7: Future Scope & Advanced Enhancements**
   * 7.1 Deep Learning (LSTM & Transformers)
   * 7.2 Live Broker APIs
   * 7.3 Multi-factor Credit Models
8. **Chapter 8: Conclusion & Bibliography**

---

## 📝 Chapter-by-Chapter Writing Guidelines

### CHAPTER 1: INTRODUCTION
* **Goal**: Define the scope of the project. Explain that while standard retail trading platforms focus on simple charts, institutional investors require **quantitative risk parameters** (VaR, GARCH, WACC, Probability of Default) to allocate capital safely.
* **Problem Statement**: Standard retail investors lack access to unified valuation and risk suites, leading to poor risk management. This project bridges that gap.

### CHAPTER 2: METHODOLOGY & THEORY
* **Formula Focus**: In this section, write out the core mathematical frameworks:
  * **ARIMA(p,d,q)**: Explain autoregressive and moving average lags, and how differencing transforms a non-stationary stock series into a stationary one.
  * **GARCH(1,1)**: Explain volatility clustering and the conditional variance equation:
    $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
  * **DCF & Gordon Growth Model**: Detail the discounting mechanism of future free cash flows and the calculation of Terminal Value:
    $$TV = \frac{FCF_n \times (1 + g)}{WACC - g}$$
  * **Monte Carlo (GBM)**: Explain stochastic brownian motion and log-normality assumptions.
  * **Value at Risk (VaR)**: Contrast Historical Simulation, Parametric VaR, and Monte Carlo VaR.
  * **Credit Risk Classifier**: Define Logistic Regression and the Sigmoid function:
    $$P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$
  * **Modern Portfolio Theory (MPT)**: Explain the mathematical search for weights that maximize the Sharpe ratio on the Efficient Frontier.

### CHAPTER 3: DATA PIPELINE
* **Pipeline description**: Document how real-time Stock OHLCV data is loaded from Yahoo Finance.
* **Caching Strategy**: Explain Streamlit’s `@st.cache_data` which caches stock arrays for 1 hour to prevent API throttling and ensure sub-second dashboard refreshes.
* **Math detail**: Explain why quantitative analysts prefer **Log Returns** over simple returns:
  $$r_t = \ln(P_t / P_{t-1})$$
  *Log returns are time-additive (the sum of daily log returns over a month equals the monthly log return), whereas simple returns are not.*

### CHAPTER 4: IMPLEMENTATION & VISUALIZATIONS
* Explain the technical reasons behind using Plotly instead of Matplotlib (interactivity, hover actions, client-side rendering, professional styling).
* Outline the CSS layout styles injected to emulate a Bloomberg Terminal (deep navy backgrounds, high-contrast text, color-coded badges for BUY/HOLD/SELL).

### CHAPTER 5: EMPIRICAL RESULTS & ANALYSIS
* **Kupiec Test**: Detail the Likelihood Ratio (LR) test output. If $LR < 3.84$, the VaR risk model is validated as statistically accurate.
* **Credit default validation**: Detail the AUC-ROC score (e.g., 0.88), showing that the Logistic Credit model has high discriminative power for corporate defaults.
* **Frontier Analysis**: Explain how PyPortfolioOpt identifies the exact weights for Tangency (Max Sharpe) allocation.

### CHAPTER 6: LIMITATIONS
* Identify that ARIMA assumes linear relationships and cannot capture sudden geopolitical or black-swan price gaps.
* GARCH(1,1) assumes symmetric volatility responses (good news and bad news spike volatility equally), whereas real markets exhibit an **asymmetry leverage effect** (bad news spikes volatility much more than good news).

### CHAPTER 7: FUTURE SCOPE
* Detail transition from statistical time-series to Deep Learning (LSTM, GRUs, or Temporal Fusion Transformers).
* Integrating natural language processing (NLP) to scrape and sentiment-score financial news (Twitter, Reddit, Bloomberg headlines) and inject sentiment as a risk factor in the credit and volatility models.

### CHAPTER 8: BIBLIOGRAPHY
* Include academic references:
  1. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*.
  2. Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*.
  3. Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*.
