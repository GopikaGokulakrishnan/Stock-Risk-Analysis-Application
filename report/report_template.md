# Risk Analytics Dashboard — Project Report
**MCA Financial Analytics Capstone 2024**

---

## 1. Introduction
This project presents a production-grade Risk Analytics Dashboard for NSE/BSE equities. It replicates institutional risk management systems used by banks and portfolio managers.

**Objective:** Build a FinTech-style interactive analytics platform using Python, Streamlit, and quantitative finance models.

---

## 2. Objectives
- Fetch and process live market data from Yahoo Finance
- Build 10 analytical modules covering forecasting, risk, and valuation
- Create a professional dark-theme dashboard with Plotly visualizations
- Demonstrate mastery of quantitative finance models in Python

---

## 3. Methodology

### Data Pipeline
- Source: Yahoo Finance via `yfinance` library
- Data: OHLCV (Open, High, Low, Close, Volume) for NSE stocks
- Processing: Log returns, forward-fill for missing values, adjusted close

### Models Used
| Module | Model | Library |
|--------|-------|---------|
| Price Forecast | ARIMA(p,d,q) | pmdarima |
| Volatility | GARCH(1,1) | arch |
| Valuation | DCF + Gordon Growth | Pure Python |
| Simulation | Geometric Brownian Motion | NumPy |
| Risk | VaR (3 methods) + CVaR | scipy |
| Credit | Logistic Regression | scikit-learn |
| Portfolio | Efficient Frontier | PyPortfolioOpt |
| Stress | Scenario Analysis | NumPy |
| Correlation | Pearson Correlation | pandas |

---

## 4. Data Pipeline

```
Yahoo Finance API
      ↓
yfinance.download()
      ↓
pandas DataFrame (OHLCV)
      ↓
Cleaning & Forward-fill
      ↓
Log Returns Computation
      ↓
Cached with @st.cache_data
      ↓
Analytics Modules
      ↓
Plotly Visualizations
      ↓
Streamlit Dashboard
```

---

## 5. Module Results

### Module 1 — Executive Summary
Aggregates KPIs from all modules into one command-centre view.
Displays investment signal (BUY/HOLD/SELL) based on Margin of Safety and VaR.

### Module 2 — ARIMA Forecasting
- auto_arima selects optimal (p,d,q) using AIC criterion
- 90-day forecast with 90% confidence intervals
- Walk-forward validation with RMSE, MAE, MAPE metrics

### Module 3 — GARCH Volatility
- GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
- Identifies volatility clustering and regime detection

### Module 4 — DCF Valuation
- Two-stage FCF projection (high growth → stable growth)
- Terminal value via Gordon Growth: TV = FCF_n × (1+g) / (WACC-g)
- Margin of Safety = (IV - Price) / IV × 100

### Module 5 — Monte Carlo
- GBM: S(t+1) = S(t) × exp((μ - 0.5σ²)Δt + σ√Δt·Z)
- 1000+ paths, probability analysis at multiple horizons

### Module 6 — Value at Risk
- Historical VaR, Parametric VaR, Monte Carlo VaR
- CVaR (Expected Shortfall) for tail risk
- Kupiec backtesting for model validation

### Module 7 — Credit Risk
- Logistic regression on 5 financial ratios
- Probability of Default mapped to credit grade (AAA–CCC)
- AUC-ROC metric for model evaluation

### Module 8 — Portfolio Optimization
- 5000 random portfolios on Efficient Frontier
- Max-Sharpe optimal weights via PyPortfolioOpt
- Pie chart allocation + Capital Market Line

### Module 9 — Stress Testing
- 5 predefined scenarios + 1 custom
- Factor sensitivity: Impact = β × Market Shock × Beta Multiplier
- Horizontal bar chart of scenario impacts

### Module 10 — Correlation Heatmap
- Pearson correlation matrix heatmap
- Rolling 60-day correlation over time
- Diversification score

---

## 6. Limitations

- ARIMA assumes linear relationships; misses structural breaks
- GARCH assumes constant model parameters over time
- DCF is highly sensitive to WACC and growth rate assumptions
- Monte Carlo uses GBM which doesn't capture market crashes (jump risk)
- Credit model uses synthetic data (no real default histories available)
- VaR underestimates tail risk during correlated market crashes
- Correlations spike toward 1.0 during crises

---

## 7. Future Scope

- LSTM/Transformer deep learning for price forecasting
- Real-time WebSocket data feed (NSE live API)
- Sentiment analysis from news and social media
- Options pricing module (Black-Scholes, Greeks)
- User authentication and portfolio tracking
- Cloud database integration (PostgreSQL/Firebase)
- Automated alerts and email notifications
- Factor models (Fama-French 3-factor)

---

## 8. Conclusion

The Risk Analytics Dashboard successfully implements 10 quantitative finance modules in a professional, modular Python architecture. The system provides actionable insights for investment decisions combining valuation (DCF), forecasting (ARIMA), risk quantification (VaR, GARCH), and portfolio management (Efficient Frontier).

The project demonstrates practical application of Financial Engineering concepts in a production-quality FinTech platform.

---

*Word Count: ~800 words (expand each section to 8–12 pages for final submission)*
