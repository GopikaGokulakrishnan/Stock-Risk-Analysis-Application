"""
Portfolio Optimization Module
-----------------------------
This module implements Modern Portfolio Theory (MPT) proposed by Harry Markowitz.
It fetches historical data for a user-defined basket of assets, computes expected 
returns and the covariance matrix, simulates 5,000 random portfolios to map the 
Efficient Frontier, and uses PyPortfolioOpt to mathematically locate:
1. The Maximum Sharpe Ratio Portfolio (Tangency Portfolio)
2. The Minimum Variance Portfolio

Visualizations:
1. Efficient Frontier Scatter Plot (Plotly color-coded by Sharpe Ratio)
2. Portfolio Allocation Pie Chart (Plotly)

Theoretical Concepts:
- Diversification: Reducing idiosyncratic risk by holding non-perfectly correlated assets.
- Efficient Frontier: Set of optimal portfolios offering highest return for a level of risk.
- Sharpe Ratio Optimization: Maximizing excess return per unit of standard deviation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns as pyp_returns
from pypfopt import risk_models as pyp_risk
from utils.helpers import get_plotly_layout
from modules.data_loader import fetch_multiple_tickers
from utils.metrics import calculate_sharpe_ratio

def run_portfolio_module(
    selected_tickers: list,
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.07
) -> tuple:
    """
    Executes Markowitz portfolio optimization, simulates the Efficient Frontier,
    solves for optimal asset weights, visualizes results, and returns optimal portfolios.
    
    Parameters:
        selected_tickers (list): List of ticker symbols to optimize.
        start_date (str): Fetch start date.
        end_date (str): Fetch end date.
        risk_free_rate (float): Annualized risk-free rate, default 7.0%.
        
    Returns:
        Tuple[float, float, dict]: (Max Sharpe Portfolio Return, Max Sharpe Portfolio Volatility, Optimal Weights Dict)
    """
    st.markdown(f"### 💼 Portfolio Optimization & Markowitz Efficient Frontier")
    
    if len(selected_tickers) < 2:
        st.warning("⚠️ Please select at least 2 tickers in the sidebar toggles to optimize a portfolio.")
        return 0.12, 0.15, {}
        
    # 1. Fetch Multilateral Data
    with st.spinner("Fetching asset history and aligning returns..."):
        prices_df, returns_df = fetch_multiple_tickers(selected_tickers, start_date, end_date)
        
    if prices_df.empty or len(prices_df.columns) < 2:
        st.error("Failed to load sufficient multilateral pricing history.")
        return 0.12, 0.15, {}
        
    n_assets = len(prices_df.columns)
    
    # 2. Compute Expected Returns and Covariance (Annualized)
    # Using PyPortfolioOpt standard estimators
    try:
        mu_ann = pyp_returns.mean_historical_return(prices_df, returns_data=False, compounding=True)
        cov_ann = pyp_risk.CovarianceShrinkage(prices_df, returns_data=False).ledoit_wolf()
    except Exception as e:
        # Fallback to simple pandas math in case of PyPortfolioOpt matrix edge-cases
        st.warning(f"Advanced covariance shrinkage failed: {str(e)}. Falling back to sample covariance.")
        mu_ann = returns_df.mean() * 252
        cov_ann = returns_df.cov() * 252

    # 3. Simulate 5,000 Random Portfolios
    n_portfolios = 5000
    results = np.zeros((3, n_portfolios))
    weights_record = []
    
    np.random.seed(42)
    for i in range(n_portfolios):
        # Generate random weights summing to 1.0
        w = np.random.random(n_assets)
        w /= np.sum(w)
        weights_record.append(w)
        
        # Portfolio Expected Return = w^T * mu
        p_ret = np.dot(w, mu_ann)
        # Portfolio Volatility = sqrt(w^T * Cov * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_ann, w)))
        # Sharpe
        p_sharpe = (p_ret - risk_free_rate) / p_vol if p_vol > 0 else 0
        
        results[0, i] = p_ret
        results[1, i] = p_vol
        results[2, i] = p_sharpe
        
    # 4. Find Mathematically Optimal Portfolios (PyPortfolioOpt Solver)
    try:
        # Max Sharpe Optimization
        ef_max = EfficientFrontier(mu_ann, cov_ann)
        raw_weights_max = ef_max.max_sharpe(risk_free_rate=risk_free_rate)
        weights_max = ef_max.clean_weights()
        ret_max, vol_max, sharpe_max = ef_max.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        
        # Min Variance Optimization
        ef_min = EfficientFrontier(mu_ann, cov_ann)
        raw_weights_min = ef_min.min_volatility()
        weights_min = ef_min.clean_weights()
        ret_min, vol_min, sharpe_min = ef_min.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        
    except Exception as e:
        st.warning(f"Mathematical solver did not converge: {str(e)}. Locating best portfolio from simulated set.")
        # Fallback to empirical search from simulated set
        max_idx = np.argmax(results[2])
        ret_max, vol_max, sharpe_max = results[0, max_idx], results[1, max_idx], results[2, max_idx]
        weights_max = dict(zip(prices_df.columns, weights_record[max_idx]))
        
        min_idx = np.argmin(results[1])
        ret_min, vol_min, sharpe_min = results[0, min_idx], results[1, min_idx], results[2, min_idx]
        weights_min = dict(zip(prices_df.columns, weights_record[min_idx]))

    # Display Performance Grid
    po_col1, po_col2, po_col3 = st.columns(3)
    with po_col1:
        st.markdown("**Maximum Sharpe Portfolio**")
        st.metric("Annualized Return", f"{ret_max*100:.2f}%")
        st.metric("Annualized Volatility (Risk)", f"{vol_max*100:.2f}%", delta=f"{sharpe_max:.2f} Sharpe")
    with po_col2:
        st.markdown("**Minimum Variance Portfolio**")
        st.metric("Annualized Return", f"{ret_min*100:.2f}%")
        st.metric("Annualized Volatility (Risk)", f"{vol_min*100:.2f}%", delta=f"{sharpe_min:.2f} Sharpe")
    with po_col3:
        st.markdown("**Optimized Weights Allocation**")
        for asset, w_val in weights_max.items():
            st.markdown(f"* **{asset}**: `{w_val*100:.1f}%` weight")

    st.markdown("---")

    # 5. Visualizations: Frontier Plot and Weight Allocation Pie Chart side-by-side
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("#### 📈 Markowitz Efficient Frontier")
        fig_frontier = go.Figure()
        
        # Plot simulated portfolios
        fig_frontier.add_trace(
            go.Scatter(
                x=results[1] * 100.0,
                y=results[0] * 100.0,
                mode='markers',
                name='Simulated Portfolios',
                marker=dict(
                    color=results[2],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio', thickness=15),
                    size=4
                ),
                hoverinfo='text',
                hovertext=[f"Ret: {r*100:.2f}%<br>Vol: {v*100:.2f}%<br>Sharpe: {s:.2f}" for r, v, s in zip(results[0], results[1], results[2])]
            )
        )
        
        # Highlight Max Sharpe Portfolio (Gold Star)
        fig_frontier.add_trace(
            go.Scatter(
                x=[vol_max * 100.0],
                y=[ret_max * 100.0],
                mode='markers',
                name='Max Sharpe Ratio',
                marker=dict(color='#FBBF24', size=15, symbol='star', line=dict(color='#000', width=1.5))
            )
        )
        
        # Highlight Min Variance Portfolio (Cyan Diamond)
        fig_frontier.add_trace(
            go.Scatter(
                x=[vol_min * 100.0],
                y=[ret_min * 100.0],
                mode='markers',
                name='Min Variance',
                marker=dict(color='#06B6D4', size=13, symbol='diamond', line=dict(color='#000', width=1.5))
            )
        )
        
        fig_frontier.update_layout(get_plotly_layout("Modern Portfolio Efficient Frontier", "Annualized Portfolio Risk (Volatility) (%)", "Annualized Expected Return (%)"))
        st.plotly_chart(fig_frontier, use_container_width=True)
        
    with g_col2:
        st.markdown("#### 🍕 Max Sharpe Weight Allocation")
        # Pie chart
        pie_labels = list(weights_max.keys())
        pie_values = list(weights_max.values())
        
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=pie_labels,
                    values=pie_values,
                    hole=0.4,
                    marker=dict(colors=['#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#3B82F6'])
                )
            ]
        )
        
        fig_pie.update_layout(get_plotly_layout("Tangency (Max Sharpe) Asset Allocation", "", ""))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # Theoretical Explanation
    with st.expander("📚 Theoretical Framework: MPT & Efficient Frontier"):
        st.markdown(r"""
        ### Modern Portfolio Theory (MPT) & Diversification
        Proposed by Harry Markowitz in 1952 (for which he won the Nobel Prize), MPT mathematically demonstrates that an investor can reduce a portfolio's overall risk simply by combining assets that are not perfectly positively correlated ($\rho < 1.0$). 
        
        * **Idiosyncratic (Specific) Risk**: Can be diversified away.
        * **Systemic (Market) Risk**: Cannot be diversified away.
        
        ### Expected Portfolio Return and Variance Formulas
        For a portfolio of $n$ assets with weights $w = [w_1, w_2, \dots, w_n]^T$ and expected returns $\mu = [\mu_1, \mu_2, \dots, \mu_n]^T$:
        $$\text{Expected Portfolio Return } E[R_p] = w^T \mu = \sum_{i=1}^n w_i \mu_i$$
        
        The portfolio variance is defined using the covariance matrix $\Sigma$:
        $$\text{Portfolio Variance } \sigma_p^2 = w^T \Sigma w = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}$$
        $$\text{Portfolio Volatility } \sigma_p = \sqrt{w^T \Sigma w}$$
        
        ### The Efficient Frontier
        The Efficient Frontier is the locus of portfolios that offer the highest expected return for a defined level of risk, or the lowest risk for a given level of expected return. Portfolios located *below* the frontier are sub-optimal because they do not offer enough return for their level of risk.
        
        ### Max Sharpe (Tangency) vs Minimum Variance Portfolios
        1. **Maximum Sharpe Portfolio (Tangency Portfolio)**:
           Locates the point on the Efficient Frontier that maximizes the Sharpe Ratio. It represents the point where a line from the risk-free rate ($R_f$) is tangent to the Efficient Frontier, maximizing the slope:
           $$\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$$
           Subject to: $\sum w_i = 1$ and $w_i \ge 0$ (no short selling).
           
        2. **Minimum Variance Portfolio**:
           Locates the portfolio with the absolute lowest risk (volatility) on the Efficient Frontier, regardless of the expected return:
           $$\min_w w^T \Sigma w$$
           Subject to: $\sum w_i = 1$ and $w_i \ge 0$.
        """)
        
    return float(ret_max), float(vol_max), weights_max
