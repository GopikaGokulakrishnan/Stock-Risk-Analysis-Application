# Institutional Risk Analytics Dashboard (Bloomberg Terminal Emulator)

A complete, production-grade, interactive financial risk management and valuation dashboard for NSE/BSE stocks. Built using Python, Streamlit, and Plotly. Designed for institutional portfolios, quantitative risk analysts, and university capstone vivas.

---

## 🏛️ Project Overview

This platform mimics institutional risk analytics suites like **Bloomberg Terminal** or **TradingView** to deliver a comprehensive 10-module quantitative workspace. It links directly to Yahoo Finance to pull real-time historical market and corporate balance sheet data, pre-processes returns, and runs advanced financial engineering models.

### Key Architecture Components
* **Dynamic Data Engine**: Automated fetching, linear interpolation of missing pricing metrics, and dual return models (Simple & Logarithmic returns).
* **Forecasting Suite**: ARIMA time-series modeling and GARCH(1,1) volatility regime trackers.
* **Valuation & Simulation**: Gordon Growth DCF Intrinsic Value waterfall charts and Geometric Brownian Motion (GBM) Monte Carlo paths.
* **Risk Control Desk**: Tri-method Value at Risk (VaR), Conditional VaR (Expected Shortfall), Kupiec POF backtesting, and macro Stress Testing.
* **Portfolio Engine**: Markowitz Efficient Frontiers and Pearson correlation diversification sweeps.

---

## 📂 Project Structure

```
risk_dashboard/
├── app.py                      # Master Streamlit dashboard entrypoint
├── requirements.txt            # Package dependencies
├── README.md                   # Project run documentation (this file)
├── PROJECT_EXPLANATION_GUIDE.md# Exhaustive 20-chapter technical handbook & viva preparation notes
├── report/
│   └── report_structure.md     # 8-12 page capstone report template
├── modules/                    # Self-contained business logic modules
│   ├── data_loader.py          # yfinance data pull and clean
│   ├── executive_summary.py    # Main KPI cards and trade signals
│   ├── arima_model.py          # Auto-ARIMA forecasting & error metrics
│   ├── garch_model.py          # Volatility modeling & regimes
│   ├── dcf_model.py            # DCF Intrinsic Value & Waterfall chart
│   ├── monte_carlo.py          # GBM price path simulator
│   ├── var_model.py            # Parametric, Hist, and MC VaR + Kupiec Test
│   ├── credit_risk.py          # Synthetic credit regression & ROC curve
│   ├── portfolio_optimization.py # Markowitz MPT Efficient Frontier
│   ├── stress_testing.py       # Stress factor return shock simulations
│   └── correlation_heatmap.py  # Pearson matrices and diversification insights
├── utils/                      # Support utilities
│   ├── helpers.py              # CSS styling & Plotly Bloomberg templates
│   └── metrics.py              # RMSE, MAE, MAPE & Sharpe Ratio formulas
└── assets/                     # Graphic resources
```

---

## 🛠️ Tech Stack & Dependencies

The system is developed entirely in Python using industry-standard libraries:
* **Frontend**: `Streamlit` (Dynamic layouts, caching, and state parameters)
* **Visualization**: `Plotly` (Annotated heatmaps, waterfall charts, interactive pathways)
* **Mathematical Operations**: `NumPy` & `SciPy` (Sigmoid maps, matrix math, probability distributions)
* **Data Processing**: `Pandas` (Timeline series alignment, pct_change maps)
* **Financial Models**:
  * `yfinance` (Real-time market and balance sheet pipeline)
  * `pmdarima` (Hyperparameter auto-optimization for ARIMA forecasting)
  * `arch` (GARCH volatility modeling)
  * `scikit-learn` (Sigmoid-based Logistic credit default classification)
  * `PyPortfolioOpt` (Markowitz quadratic solver optimizations)

---

## 🚀 Installation & Local Execution

### 1. Prerequisite Setup
Ensure Python 3.9, 3.10, or 3.11 is installed on your machine.

### 2. Install Dependencies
Open your command terminal, navigate to the project directory, and run:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
Fire up the local host web server by executing:
```bash
streamlit run app.py
```
The dashboard will compile and open automatically in your web browser at `http://localhost:8501`.

---

## 🔧 Recent Code Update Summary
The latest revision includes the following stability and runtime fixes:
* `app.py` now wraps the Streamlit dashboard logic inside `main()` and uses `if __name__ == "__main__": main()` so the app no longer executes on import.
* `modules/data_loader.py` was improved to handle Yahoo Finance price column selection more robustly and avoid `Adj Close` related runtime failures.
* `modules/arima_model.py` now supports both `pmdarima` and native `statsmodels` forecast APIs, with a fallback path to prevent errors when one API is unavailable.
* Added runtime guards so the dashboard fails gracefully if market data cannot be loaded, improving user-facing stability.

---

## 🎓 Mathematical Formulas Applied

* **Log Returns**: $r_t = \ln(P_t / P_{t-1})$
* **GARCH(1,1)**: $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$
* **Gordon Growth Terminal Value**: $TV = \frac{FCF_n \times (1 + g_{\text{terminal}})}{WACC - g_{\text{terminal}}}$
* **Geometric Brownian Motion (GBM)**: $S_{t+1} = S_t \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z_t\right)$
* **Sharpe Ratio**: $\text{Sharpe} = \frac{E[R_p] - R_f}{\sigma_p}$
* **Kupiec Backtest LR**: $LR = -2 \ln \left( \frac{(1 - p)^{T-N} p^N}{(1 - \pi)^{T-N} \pi^N} \right) \sim \chi^2(1)$

---

## ☁️ Production Deployment

### Streamlit Cloud Deployment
1. Upload the entire project directory (`D:\risk_dashboard\`) to a public repository on **GitHub**.
2. Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New App", select your repository, specify the branch (usually `main`), and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will parse `requirements.txt`, install dependencies, and host your terminal on a public URL.
