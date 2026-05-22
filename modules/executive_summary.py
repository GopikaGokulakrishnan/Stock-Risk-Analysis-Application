"""
Executive Summary Module
------------------------
This module compiles institutional-grade KPI cards, mini sparkline charts,
an algorithmic investment signal, and a risk level indicator.

It explains:
1. Why Value at Risk (VaR) is used in asset management.
2. What the Sharpe Ratio measures.
3. The definition and utility of the Margin of Safety.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
from utils.helpers import create_kpi_card, inject_bloomberg_css

def run_executive_summary_module(
    ticker: str,
    df: pd.DataFrame,
    forecasted_price: float,
    intrinsic_value: float,
    expected_return: float,
    portfolio_return: float,
    portfolio_risk: float,
    var_95: float,
    prob_default: float,
    sharpe_ratio: float
) -> None:
    """
    Renders the executive summary dashboard tab with dynamic KPIs, sparkline,
    and a structured overview of risk signals.
    
    Parameters:
        ticker (str): Stock ticker.
        df (pd.DataFrame): Stock historical dataframe.
        forecasted_price (float): 90-day forecasted stock price.
        intrinsic_value (float): DCF intrinsic stock price.
        expected_return (float): Annualized expected return.
        portfolio_return (float): Optimized portfolio annualized return.
        portfolio_risk (float): Optimized portfolio annualized volatility.
        var_95 (float): 95% 1-day Value at Risk (usually expressed as a negative %).
        prob_default (float): Credit risk Probability of Default (%).
        sharpe_ratio (float): Sharpe ratio of the stock.
    """
    inject_bloomberg_css()
    
    current_price = df['Adj Close'].iloc[-1]
    
    # Calculate Margin of Safety
    # MOS = (Intrinsic Value - Current Price) / Intrinsic Value
    if intrinsic_value > 0:
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
    else:
        margin_of_safety = 0.0
        
    # Algorithmic Investment Signal
    # BUY: Margin of Safety > 20% AND VaR > -5% (i.e. daily tail risk is less than 5% loss)
    # HOLD: Margin of Safety between 5% and 20%
    # SELL: Otherwise
    var_percentage = var_95 * 100 # Convert to percentage e.g. -0.035 -> -3.5%
    mos_percentage = margin_of_safety * 100
    
    if mos_percentage > 20.0 and var_percentage > -5.0:
        signal = "BUY"
        signal_badge = f'<span class="badge-buy">STRONG BUY</span>'
        signal_color = "#10B981"
    elif 5.0 <= mos_percentage <= 20.0:
        signal = "HOLD"
        signal_badge = f'<span class="badge-hold">ACCUMULATE / HOLD</span>'
        signal_color = "#94A3B8"
    else:
        signal = "SELL"
        signal_badge = f'<span class="badge-sell">REDUCE / SELL</span>'
        signal_color = "#EF4444"
        
    # Risk Level Indicator
    if var_percentage < -4.0 or prob_default > 10.0:
        risk_level = "HIGH RISK"
        risk_color = "#EF4444"
    elif var_percentage < -2.0 or prob_default > 2.0:
        risk_level = "MODERATE RISK"
        risk_color = "#F59E0B"
    else:
        risk_level = "LOW RISK"
        risk_color = "#10B981"

    st.markdown(f"### 📊 Institutional Executive Summary: {ticker}")
    
    # Grid of Metric Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            create_kpi_card(
                "Current Market Price", 
                f"₹ {current_price:,.2f}", 
                "Live Yahoo Finance Fetch"
            ), 
            unsafe_allow_html=True
        )
        st.markdown(
            create_kpi_card(
                "DCF Intrinsic Value", 
                f"₹ {intrinsic_value:,.2f}", 
                f"Margin of Safety: {mos_percentage:.2f}%"
            ), 
            unsafe_allow_html=True
        )
        st.markdown(
            create_kpi_card(
                "Expected Annual Return", 
                f"{expected_return * 100:.2f}%", 
                "Based on log-returns mean"
            ), 
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            create_kpi_card(
                "90-Day ARIMA Forecast", 
                f"₹ {forecasted_price:,.2f}", 
                f"Projected trend change"
            ), 
            unsafe_allow_html=True
        )
        st.markdown(
            create_kpi_card(
                "Value at Risk (VaR 95%)", 
                f"{var_percentage:.2f}%", 
                "Maximum daily historical loss"
            ), 
            unsafe_allow_html=True
        )
        st.markdown(
            create_kpi_card(
                "Probability of Default (PD)", 
                f"{prob_default:.2f}%", 
                "Credit rating model output"
            ), 
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            create_kpi_card(
                "Optimized Portfolio Return", 
                f"{portfolio_return * 100:.2f}%", 
                "Max Sharpe Allocation"
            ), 
            unsafe_allow_html=True
        )
        st.markdown(
            create_kpi_card(
                "Optimized Portfolio Risk", 
                f"{portfolio_risk * 100:.2f}%", 
                "Annualized Volatility"
            ), 
            unsafe_allow_html=True
        )
        st.markdown(
            create_kpi_card(
                "Sharpe Ratio", 
                f"{sharpe_ratio:.2f}", 
                "Risk-adjusted excess return"
            ), 
            unsafe_allow_html=True
        )
        
    # Signal and Sparkline section
    grid_col1, grid_col2 = st.columns([1, 1])
    
    with grid_col1:
        st.markdown("#### 🚨 Algorithmic Trade Signals")
        signal_html = f"""
        <div style="background: #11151F; border: 1px solid #1E293B; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Consensus Recommendation</div>
            <div style="margin-bottom: 1rem;">{signal_badge}</div>
            <div style="font-size: 0.9rem; color: #94A3B8;">Asset Risk Class: <b style="color: {risk_color}">{risk_level}</b></div>
            <div style="font-size: 0.75rem; color: #475569; margin-top: 0.5rem; text-align: left; line-height: 1.3;">
                * Signal Conditions:<br/>
                - BUY: Margin of Safety > 20% AND VaR > -5.0% (low tail volatility)<br/>
                - HOLD: Margin of Safety between 5.0% and 20.0%<br/>
                - SELL: Otherwise.
            </div>
        </div>
        """
        st.markdown(signal_html, unsafe_allow_html=True)
        
    with grid_col2:
        st.markdown("#### 📉 60-Day Historic Trend (Sparkline)")
        spark_data = df['Adj Close'].iloc[-60:]
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=spark_data.index,
                y=spark_data.values,
                mode='lines',
                line=dict(color=signal_color, width=3.5),
                fill='tozeroy',
                fillcolor=f'rgba({",".join([str(int(c*255)) for c in hex_to_rgb(signal_color)])}, 0.1)'
            )
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=130,
            paper_bgcolor="#11151F",
            plot_bgcolor="#11151F",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
    st.markdown("---")
    
    # Financial Theory Quick Reference (Viva prep)
    with st.expander("🎓 Technical Viva Explanations (Formulas & Concepts)"):
        st.markdown("""
        ### 1. Value at Risk (VaR)
        * **Why it is used**: VaR summarizes the worst expected loss over a specific horizon (e.g., 1 day) at a given confidence level (e.g., 95%) under normal market conditions. It translates statistical distribution into a single, intuitive cash or percentage threshold for asset managers and risk committees.
        * **Formula**: 
          $$\\text{VaR}_\\alpha = \\inf \\{ L \\mid P(L > L_0) \\le 1 - \\alpha \\}$$
          
        ### 2. Sharpe Ratio
        * **What it means**: Developed by William F. Sharpe, it measures the excess return per unit of volatility in an investment asset. It tells us whether an asset's excess returns are due to smart investment decisions or excessive leverage and volatility.
        * **Formula**:
          $$\\text{Sharpe Ratio} = \\frac{E[R_p] - R_f}{\\sigma_p}$$
          where $E[R_p]$ is annualized expected return, $R_f$ is the risk-free rate, and $\\sigma_p$ is the annualized standard deviation.
          
        ### 3. Margin of Safety (MoS)
        * **What it means**: Made famous by Benjamin Graham and Warren Buffett, it represents the percentage discount of the current stock price relative to its intrinsic value. A high margin of safety protects investor capital against errors in forecasting or unexpected market stress.
        * **Formula**:
          $$\\text{Margin of Safety} = \\frac{\\text{Intrinsic Value} - \\text{Market Price}}{\\text{Intrinsic Value}}$$
        """)
