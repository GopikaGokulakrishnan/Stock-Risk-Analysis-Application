"""
Monte Carlo Simulation Module
-----------------------------
This module implements stochastic simulations for asset pricing using Geometric 
Brownian Motion (GBM).

Stochastic Differential Equation (SDE):
    dS_t = mu * S_t * dt + sigma * S_t * dW_t
    
Discrete Solution (GBM Equation):
    S(t+1) = S(t) * exp((mu - 0.5 * sigma^2)*dt + sigma * sqrt(dt) * Z)
    where Z ~ N(0, 1)

Theoretical Concepts:
1. Stochastic Processes: Systems characterized by random variables evolving over time.
2. GBM Assumptions: Constant drift/volatility, log-normal distributions, efficient markets.
3. Market Randomness: Integrating stochastic brownian motion shocks to model fat-tails/risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import get_plotly_layout
import time

def run_monte_carlo_module(
    ticker: str,
    df: pd.DataFrame,
    num_paths: int = 1000,
    forecast_days: int = 90,
    target_upside: float = 1.15,  # 15% upside target
    stop_loss_downside: float = 0.85  # 15% stop-loss
) -> tuple:
    """
    Runs the GBM Monte Carlo simulation, tracks execution speed, plots interactive
    simulated pathways and terminal distributions, and displays probability metrics.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        df (pd.DataFrame): Stock historical price dataframe.
        num_paths (int): Number of simulated price pathways (e.g., 1000, 5000).
        forecast_days (int): Number of days to forecast out (e.g. 90).
        target_upside (float): Ratio multiplier for target price.
        stop_loss_downside (float): Ratio multiplier for stop-loss price.
        
    Returns:
        Tuple[float, np.ndarray]: Median terminal price, and full array of terminal prices.
    """
    st.markdown(f"### 🎲 Stochastic Monte Carlo Simulation: {ticker}")
    
    prices = df['Adj Close']
    returns = df['Log_Return'].dropna()
    
    if len(prices) < 30:
        st.warning("Insufficient history for setting simulation parameters.")
        return prices.iloc[-1], np.array([prices.iloc[-1]])
        
    # Calculate historical parameters (daily scale)
    mu = float(returns.mean())
    sigma = float(returns.std())
    start_price = float(prices.iloc[-1])
    
    dt = 1.0 # Daily time step (1 day)
    
    # Track execution time
    t0 = time.time()
    
    # 1. Vectorized GBM Simulation
    # S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    drift = mu - 0.5 * (sigma ** 2)
    
    # Random normal shocks: matrix of shape (forecast_days, num_paths)
    Z = np.random.standard_normal((forecast_days, num_paths))
    
    # Vectorized computation of path returns
    shock_returns = np.exp(drift * dt + sigma * np.sqrt(dt) * Z)
    
    # Accumulate returns (cumulative product over time steps)
    price_paths = np.zeros((forecast_days + 1, num_paths))
    price_paths[0, :] = start_price
    
    for t in range(1, forecast_days + 1):
        price_paths[t, :] = price_paths[t - 1, :] * shock_returns[t - 1, :]
        
    comp_time_ms = (time.time() - t0) * 1000.0
    
    # 2. Extract Terminal Metrics
    terminal_prices = price_paths[-1, :]
    median_terminal = float(np.median(terminal_prices))
    mean_terminal = float(np.mean(terminal_prices))
    p10_terminal = float(np.percentile(terminal_prices, 10))
    p90_terminal = float(np.percentile(terminal_prices, 90))
    
    # Target and stop-loss levels
    target_price = start_price * target_upside
    stop_loss_price = start_price * stop_loss_downside
    
    # Probabilities
    prob_exceed_target = float(np.sum(terminal_prices >= target_price) / num_paths) * 100
    prob_hit_stop = float(np.sum(terminal_prices <= stop_loss_price) / num_paths) * 100
    
    # Display Grid
    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
    with mc_col1:
        st.metric("Median Terminal Price", f"₹ {median_terminal:,.2f}", 
                  help="50th percentile of all simulated paths")
    with mc_col2:
        st.metric("Probability of Exceeding Target", f"{prob_exceed_target:.2f}%",
                  help=f"Target Price: ₹{target_price:.2f} ({target_upside*100-100:.0f}% upside)")
    with mc_col3:
        st.metric("Probability of Hitting Stop-Loss", f"{prob_hit_stop:.2f}%",
                  help=f"Stop-Loss: ₹{stop_loss_price:.2f} ({100-stop_loss_downside*100:.0f}% downside)")
    with mc_col4:
        st.metric("Computation Speed", f"{comp_time_ms:.2f} ms", 
                  help=f"Simulated {num_paths} paths over {forecast_days} periods")
        
    # Render Path and Distribution Charts side-by-side
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("#### 📈 Simulated Price Pathways")
        # Plotly lines chart (downsample paths to max 100 to keep browser responsive)
        fig_paths = go.Figure()
        
        timeline = list(range(forecast_days + 1))
        max_plot_paths = min(100, num_paths)
        
        for i in range(max_plot_paths):
            fig_paths.add_trace(
                go.Scatter(
                    x=timeline,
                    y=price_paths[:, i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(94, 234, 212, 0.2)'), # Muted Cyan
                    showlegend=False
                )
            )
            
        # Add Median, 10th and 90th percentile paths
        median_path = np.percentile(price_paths, 50, axis=1)
        p10_path = np.percentile(price_paths, 10, axis=1)
        p90_path = np.percentile(price_paths, 90, axis=1)
        
        fig_paths.add_trace(go.Scatter(x=timeline, y=median_path, mode='lines', name='Median Path (50th Pctl)', line=dict(color='#06B6D4', width=3)))
        fig_paths.add_trace(go.Scatter(x=timeline, y=p90_path, mode='lines', name='Bull Case (90th Pctl)', line=dict(color='#10B981', width=2, dash='dash')))
        fig_paths.add_trace(go.Scatter(x=timeline, y=p10_path, mode='lines', name='Bear Case (10th Pctl)', line=dict(color='#EF4444', width=2, dash='dash')))
        
        fig_paths.update_layout(get_plotly_layout("GBM Random Asset Pathways", "Forecast Days", "Price (₹)"))
        st.plotly_chart(fig_paths, use_container_width=True)
        
    with g_col2:
        st.markdown("#### 📊 Terminal Price Distribution")
        # Plotly Histogram/Probability Density
        fig_dist = go.Figure()
        
        fig_dist.add_trace(
            go.Histogram(
                x=terminal_prices,
                nbinsx=40,
                name='Terminal Price Frequency',
                marker_color='#F59E0B',
                opacity=0.75,
                histnorm='probability'
            )
        )
        
        # Vertical indicators
        fig_dist.add_vline(x=start_price, line_width=2, line_dash="dash", line_color="#94A3B8", annotation_text="Start Price")
        fig_dist.add_vline(x=target_price, line_width=2, line_color="#10B981", annotation_text="Target Price")
        fig_dist.add_vline(x=stop_loss_price, line_width=2, line_color="#EF4444", annotation_text="Stop-Loss")
        
        fig_dist.update_layout(get_plotly_layout("Probability Distribution at Horizon", "Terminal Stock Price (₹)", "Probability density"))
        st.plotly_chart(fig_dist, use_container_width=True)
        
    # Theoretical Explanation
    with st.expander("📚 Theoretical Framework: Stochastic Simulations & GBM"):
        st.markdown(r"""
        ### What are Stochastic Simulations?
        In finance, asset prices cannot be forecasted with absolute certainty due to systemic and idiosyncratic shocks. A **stochastic simulation** models asset prices as dynamic processes driven by deterministic rules (drift) combined with random shocks (volatility). By running thousands of parallel paths, we create a **probability distribution** of future prices rather than a single static forecast.
        
        ### Geometric Brownian Motion (GBM) Formula
        GBM is the standard mathematical model used to describe stock price behavior (and forms the basis of the Black-Scholes option pricing model). The continuous-time stochastic differential equation is:
        $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
        
        Where:
        * **$S_t$**: The stock price at time $t$.
        * **$\mu$ (Drift)**: The expected rate of return of the stock (deterministic trend).
        * **$\sigma$ (Volatility)**: The standard deviation of stock returns, representing risk.
        * **$dW_t$**: A standard Wiener process (Brownian motion shock) representing random market news.
        
        By applying **Ito's Lemma** to solve this SDE, we get the discrete-time formula used in our code:
        $$S_{t+1} = S_t \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z_t\right)$$
        where $Z_t \sim \mathcal{N}(0, 1)$ represents a standard normal random shock for day $t$.
        
        ### GBM Model Assumptions
        1. **Log-Normality**: The returns of the stock are normally distributed, which implies stock prices are log-normally distributed (prices can never drop below zero, preventing negative valuations).
        2. **Constant Drift & Volatility**: The parameters $\mu$ and $\sigma$ are assumed to be constant throughout the simulation window (which is a limitation, as real markets experience volatility shifts, as shown in the GARCH model).
        3. **No Arbitrage**: Markets are efficient, and prices adjust instantly to news (represented by the independent normal random shocks $Z_t$).
        """)
        
    return median_terminal, terminal_prices
