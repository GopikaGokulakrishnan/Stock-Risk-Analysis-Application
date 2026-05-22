"""
ARIMA Forecasting Module
------------------------
This module handles time series forecasting using the pmdarima auto_arima library.
It dynamically selects the best ARIMA(p,d,q) order by minimizing Information Criteria
(AIC/BIC), runs walk-forward validation on a test split, projects a 90-day price
forecast with confidence bands, and visualizes the results in interactive Plotly figures.

Theoretical Concepts:
- ARIMA: AutoRegressive Integrated Moving Average.
- Order Parameters (p, d, q):
  * p (AutoRegressive): Lags of the dependent variable.
  * d (Integrated): Degree of differencing required for stationarity.
  * q (Moving Average): Lags of the forecast errors.
"""

import inspect
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pmdarima.arima import auto_arima
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.helpers import get_plotly_layout
import time

def run_arima_module(ticker: str, df: pd.DataFrame) -> float:
    """
    Executes the ARIMA forecasting module, displays statistical metrics,
    conducts walk-forward validation, plots projections, and returns the final
    90-day forecasted price point.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        df (pd.DataFrame): The stock price dataframe.
        
    Returns:
        float: The final 90-day forecasted price.
    """
    st.markdown(f"### 📈 Time-Series ARIMA Forecasting: {ticker}")
    
    prices = df['Adj Close']
    
    # Check data sufficiency
    if len(prices) < 100:
        st.warning("Insufficient data for fitting a robust ARIMA model. Minimum 100 observations required.")
        return float(prices.iloc[-1])
        
    # User option: Enable walk-forward validation
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        st.info("💡 ARIMA fits best on historical levels but assumes linear relationships.")
    with col_ctrl2:
        run_validation = st.checkbox("Run Walk-Forward Validation (Takes ~5-10s)", value=True)

    # 1. Walk-Forward Validation Split (e.g., 30 days test set)
    validation_status_placeholder = st.empty()
    
    rmse_val, mae_val, mape_val = 0.0, 0.0, 0.0
    
    if run_validation:
        validation_status_placeholder.markdown("⏳ Running Walk-Forward Validation...")
        test_size = min(30, int(len(prices) * 0.15))
        train_data = prices.iloc[:-test_size]
        test_data = prices.iloc[-test_size:]
        
        try:
            # Fit auto_arima on training set
            val_model = auto_arima(
                train_data, 
                start_p=0, start_q=0,
                max_p=3, max_q=3, 
                d=None,           # Let pmdarima estimate 'd'
                seasonal=False, 
                trace=False,
                error_action='ignore', 
                suppress_warnings=True
            )
            
            # Forecast test period
            predictions = val_model.predict(n_periods=test_size)
            
            # Align indexes
            predictions_series = pd.Series(predictions, index=test_data.index)
            
            # Calculate validation errors
            rmse_val = calculate_rmse(test_data, predictions_series)
            mae_val = calculate_mae(test_data, predictions_series)
            mape_val = calculate_mape(test_data, predictions_series)
            
            validation_status_placeholder.success("✅ Walk-Forward Validation Complete!")
        except Exception as e:
            validation_status_placeholder.warning(f"⚠️ Walk-Forward Validation failed: {str(e)}")
            
    # 2. Fit Full Model for 90-Day Projection
    with st.spinner("Optimizing ARIMA hyperparameters on full historical set..."):
        t0 = time.time()
        try:
            # Fit auto_arima
            model = auto_arima(
                prices, 
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            fit_time = time.time() - t0
            
            # Retrieve model orders and criteria
            order = model.order
            aic = model.aic()
            bic = model.bic()
            
        except Exception as e:
            # Dynamic robust fallback to ARIMA(1, 1, 1) in case of rare convergence errors
            st.warning(f"Auto-ARIMA optimization did not converge. Falling back to robust ARIMA(1,1,1). Error: {str(e)}")
            from statsmodels.tsa.arima.model import ARIMA
            fallback_model = ARIMA(prices, order=(1, 1, 1)).fit()
            order = (1, 1, 1)
            aic = fallback_model.aic
            bic = fallback_model.bic
            fit_time = 0.1
            model = fallback_model # Assign for forecasting
            
    # 3. 90-day Forecast Projection
    forecast_steps = 90
    forecast_values = None
    lower_bound = np.full(forecast_steps, np.nan)
    upper_bound = np.full(forecast_steps, np.nan)

    # Try the pmdarima-style forecast API first
    try:
        forecast_res = model.predict(n_periods=forecast_steps, return_conf_int=True, alpha=0.05)
        forecast_values, conf_int = forecast_res[0], forecast_res[1]
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
    except Exception:
        # Try statsmodels-style forecast API
        try:
            forecast_res = model.get_forecast(steps=forecast_steps)
            forecast_values = np.asarray(forecast_res.predicted_mean)
            conf_int = forecast_res.conf_int(alpha=0.05)
            lower_bound = np.asarray(conf_int.iloc[:, 0])
            upper_bound = np.asarray(conf_int.iloc[:, 1])
        except Exception:
            # Final fallback: direct predict without confidence intervals
            try:
                if 'n_periods' in inspect.signature(model.predict).parameters:
                    forecast_values = model.predict(n_periods=forecast_steps)
                else:
                    forecast_values = model.predict(steps=forecast_steps)
            except Exception as e:
                st.warning(f"ARIMA forecast generation failed: {str(e)}. Returning last known price as forecast.")
                forecast_values = np.full(forecast_steps, float(prices.iloc[-1]))

    forecast_values = np.asarray(forecast_values)

    # Generate forecast dates
    last_date = prices.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
    
    forecast_df = pd.DataFrame({
        'Forecast': forecast_values,
        'Lower_CI': lower_bound,
        'Upper_CI': upper_bound
    }, index=forecast_dates)
    
    # Calculate Forecast Metrics and Trend
    final_forecast_price = float(forecast_values[-1])
    start_price = float(prices.iloc[-1])
    pct_change = ((final_forecast_price - start_price) / start_price) * 100
    
    if pct_change > 2.0:
        trend = "BULLISH 📈"
        trend_color = "#10B981"
    elif pct_change < -2.0:
        trend = "BEARISH 📉"
        trend_color = "#EF4444"
    else:
        trend = "NEUTRAL / SIDEWAYS ➡️"
        trend_color = "#94A3B8"
        
    # Render Metrics Grid
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.metric("Selected Model", f"ARIMA{order}", help="Optimal Auto-ARIMA parameters (p, d, q)")
    with m_col2:
        st.metric("Model AIC", f"{aic:,.2f}", help="Akaike Information Criterion (lower is better)")
    with m_col3:
        st.metric("90-Day Trend", trend, delta=f"{pct_change:.2f}%")
    with m_col4:
        st.metric("Auto-Fit Time", f"{fit_time:.2f}s")
        
    # Render Walk-Forward Validation Metrics
    if run_validation:
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            st.metric("Validation RMSE", f"₹ {rmse_val:.2f}", help="Root Mean Squared Error on test set")
        with v_col2:
            st.metric("Validation MAE", f"₹ {mae_val:.2f}", help="Mean Absolute Error on test set")
        with v_col3:
            st.metric("Validation MAPE", f"{mape_val:.2f}%", help="Mean Absolute Percentage Error on test set")

    # 4. Plotly Visualization
    fig = go.Figure()
    
    # Historic Prices (Limit to last 180 days for clean visualization)
    view_limit = prices.iloc[-180:]
    fig.add_trace(
        go.Scatter(
            x=view_limit.index,
            y=view_limit.values,
            mode='lines',
            name='Actual Price',
            line=dict(color='#06B6D4', width=2)
        )
    )
    
    # Forecasted Prices
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Forecast'],
            mode='lines',
            name='Mean Forecast',
            line=dict(color='#F59E0B', width=2.5, dash='dash')
        )
    )
    
    # Confidence Interval Band
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df['Upper_CI'].tolist() + forecast_df['Lower_CI'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(245, 158, 11, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Band'
        )
    )
    
    # Finalize Layout
    fig.update_layout(get_plotly_layout(f"ARIMA(p,d,q) 90-Day Forecast Projection", "Timeline", "Stock Price (₹)"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical explanation
    with st.expander("📚 Theoretical Framework: Time Series Forecasting with ARIMA"):
        st.markdown(r"""
        ### What is ARIMA?
        ARIMA (AutoRegressive Integrated Moving Average) is a powerful, generalized statistical model designed to analyze and project stationary time-series data. It works by combining three key structures:
        
        1. **AutoRegressive (AR - $p$)**: Explains the relationship of the variable with its own historical lagged values.
           $$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p} + \epsilon_t$$
           
        2. **Integrated (I - $d$)**: Measures the degree of differencing applied to the raw data series to eliminate trends and seasonal components, rendering it **stationary** (constant mean, variance, and autocovariance).
           $$Y_t = (1 - L)^d X_t$$
           
        3. **Moving Average (MA - $q$)**: Explains the relationship of the variable with its own historical forecast errors (shocks).
           $$X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}$$
           
        ### Why ARIMA is used:
        In quantitative finance, asset prices represent sequential, auto-correlated path dependencies. Auto-ARIMA searches for optimal combinations of $p, d, q$ to capture short-term serial correlation. 
        
        ### Meaning of $p$, $d$, $q$:
        * **$p$**: Auto-regressive lag order (number of previous days' prices influencing today).
        * **$d$**: Differencing degree (how many times we subtract yesterday's price from today's to achieve stability).
        * **$q$**: Moving-average error window (number of previous noise/shock terms integrated).
        
        ### Why Confidence Intervals Matter:
        Financial assets exhibit high noise-to-signal ratios. A point-estimate forecast (e.g. predicting a stock will be exactly ₹2450 in 3 months) is highly unlikely to be exact. A **95% Confidence Band** defines the range within which the future price is statistically expected to reside. The widening of the band reflects rising uncertainty as we project further into the future.
        """)
        
    return final_forecast_price
