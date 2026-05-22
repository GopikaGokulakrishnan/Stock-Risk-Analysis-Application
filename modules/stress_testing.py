"""
Stress Testing Module
---------------------
This module implements macro stress testing scenarios and shock modeling for a stock.
It projects changes in asset returns, volatilities, DCF valuations, and VaR under:
1. Market Crash (-30% shock)
2. Interest Rate Hike (+2% WACC discount shock)
3. Corporate Recession (-20% FCF cash flow contraction)
4. Oil Price Shock (Beta-weighted return shock)
5. Best Case (+15% returns expansion)
6. Custom Scenario (Interactive user slider inputs)

Theoretical Concepts:
- Stress Testing: Evaluating tail risk vulnerabilities to extreme, hypothetical events.
- Scenario Analysis: Modeling historical or forward-looking macro economic shocks.
- Factor Exposure: Estimating sensitivity of assets to systemic macro variables (rates, GDP).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import get_plotly_layout

def run_stress_testing_module(
    ticker: str,
    current_price: float,
    expected_return: float,
    historical_vol: float,
    dcf_intrinsic: float,
    historical_var: float
) -> None:
    """
    Simulates macro stress scenarios, computes percentage changes, displays an
    impact table, plots a horizontal comparison bar chart, and provides risk interpretations.
    
    Parameters:
        ticker (str): Stock ticker.
        current_price (float): Current asset price.
        expected_return (float): Base expected annualized return.
        historical_vol (float): Base GARCH or rolling annualized volatility.
        dcf_intrinsic (float): Base DCF intrinsic value.
        historical_var (float): Base historical VaR (daily ratio).
    """
    st.markdown(f"### ⚡ Institutional Macro Stress Testing: {ticker}")
    
    # Custom Scenario Controls
    st.markdown("#### 🛠️ Custom Scenario Modulators")
    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        custom_return_shock = st.slider("Simulate Return Shock (%)", -50.0, 50.0, -10.0, step=1.0)
    with sc_col2:
        custom_vol_shock = st.slider("Simulate Volatility Spike (%)", -30.0, 100.0, 30.0, step=5.0)
    with sc_col3:
        custom_dcf_shock = st.slider("Simulate FCF Growth Shock (%)", -50.0, 50.0, -15.0, step=1.0)

    # Scenarios Definitions:
    # Format: [Scenario Name, Return Shock (%), Volatility Spike (%), DCF Value Shock (%)]
    scenarios = [
        ["Baseline (Normal)", 0.0, 0.0, 0.0],
        ["Market Crash (Systemic)", -30.0, 50.0, -25.0],
        ["Interest Rate Hike", -5.0, 10.0, -20.0], # Discount rate up reduces DCF
        ["Recession (FCF Contraction)", -15.0, 25.0, -35.0],
        ["Oil Price Shock", -12.0, 15.0, -10.0],
        ["Best Case Scenario", 15.0, -15.0, 20.0],
        ["Custom Scenario", custom_return_shock, custom_vol_shock, custom_dcf_shock]
    ]
    
    scenario_names = []
    projected_returns = []
    projected_vols = []
    projected_dcf = []
    projected_var = []
    
    for sc in scenarios:
        name, r_shock, v_shock, dcf_shock = sc
        scenario_names.append(name)
        
        # 1. Projected Return
        # Expected return is annualized. Shock is applied directly
        proj_r = expected_return + (r_shock / 100.0)
        projected_returns.append(proj_r)
        
        # 2. Projected Volatility
        # Volatility spike is relative e.g., 50% spike on 20% vol is 20 * (1 + 0.5) = 30%
        proj_v = historical_vol * (1.0 + (v_shock / 100.0))
        proj_v = max(0.05, proj_v) # limit bounds
        projected_vols.append(proj_v)
        
        # 3. Projected DCF Intrinsic Value
        # DCF value shock is direct percentage change
        proj_dcf = dcf_intrinsic * (1.0 + (dcf_shock / 100.0))
        proj_dcf = max(0.0, proj_dcf)
        projected_dcf.append(proj_dcf)
        
        # 4. Projected VaR
        # Standard parametric approximation of VaR change: higher vol increases VaR (makes it more negative)
        # VaR = mean_daily - Z * std_daily
        # Daily return mean = proj_r / 252. Daily vol = proj_v / sqrt(252).
        z_95 = 1.6449
        proj_var_daily = (proj_r / 252.0) - z_95 * (proj_v / np.sqrt(252.0))
        projected_var.append(proj_var_daily)
        
    # Compile Impact DataFrame
    impact_df = pd.DataFrame({
        "Scenario Name": scenario_names,
        "Projected Return (Annual)": [f"{r*100:.2f}%" for r in projected_returns],
        "Projected Volatility": [f"{v*100:.2f}%" for v in projected_vols],
        "Projected DCF Value": [f"₹ {d:,.2f}" for d in projected_dcf],
        "Projected 1-Day VaR (95%)": [f"{v*100:.2f}%" for v in projected_var]
    })
    
    # 5. Render Comparison Matrix
    st.markdown("#### 📊 Scenario Impact Comparison Matrix")
    st.table(impact_df)
    
    # 6. Plotly Horizontal Comparison Bar Chart
    # Plot expected return shifts across scenarios
    fig = go.Figure()
    
    # Map colors
    colors = []
    for r in projected_returns:
        if r > expected_return + 0.02:
            colors.append('#10B981') # Green for positive shift
        elif r < expected_return - 0.02:
            colors.append('#EF4444') # Red for negative shift
        else:
            colors.append('#64748B') # Muted slate for baseline/neutral
            
    fig.add_trace(
        go.Bar(
            y=scenario_names,
            x=[r * 100.0 for r in projected_returns],
            orientation='h',
            marker_color=colors,
            text=[f"{r*100:.1f}%" for r in projected_returns],
            textposition='outside',
            name='Projected Annual Return (%)'
        )
    )
    
    # Add vertical line at baseline return
    fig.add_vline(x=expected_return * 100.0, line_width=2, line_dash="dash", line_color="#06B6D4", annotation_text="Baseline")
    
    fig.update_layout(get_plotly_layout("Annual Expected Return Shifts Across Stress Scenarios", "Expected Return (%)", ""))
    # Adjust margins to fit long labels
    fig.update_layout(margin=dict(l=180, r=40, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)
    
    # Dynamic Risk Interpretation Text
    st.markdown("#### 🔍 Institutional Risk Interpretations")
    worst_idx = np.argmin(projected_returns)
    worst_scenario = scenario_names[worst_idx]
    worst_return = projected_returns[worst_idx] * 100
    worst_var_val = projected_var[worst_idx] * 100
    
    st.markdown(f"""
    * **Severe Stress Scenario**: The **{worst_scenario}** represents the most adverse scenario, projecting an expected return contraction down to **{worst_return:.2f}%** and triggering a daily tail risk spike (**VaR 95%**) of **{worst_var_val:.2f}%**.
    * **Capital Adequacy Buffer**: Under the **Interest Rate Hike** scenario (simulating RBI/Fed tightening), WACC discount hikes reduce intrinsic valuations by **20%**. This indicates a high exposure to interest rate risk, necessitating a hedging strategy (e.g., interest rate swaps).
    """)
    
    # Theoretical Explanation
    with st.expander("📚 Theoretical Framework: Stress Testing & Scenario Analysis"):
        st.markdown(r"""
        ### What is Stress Testing in Finance?
        While statistical models (like VaR) are excellent for predicting risk under *normal* market conditions, they frequently fail during structural market breaks or black-swan crises because historical parameters shift instantly.
        
        **Stress Testing** is a forward-looking risk management technique used by institutions and central banks (such as the Federal Reserve's CCAR audits) to evaluate the **capital adequacy** and solvency of portfolios under extreme, hypothetical shocks.
        
        ### Scenario Analysis: Historical vs. Hypothetical
        1. **Historical Scenarios**: Modeling exact replications of past structural crises:
           * *2008 Great Financial Crisis*: Equities drop 50%, credit spreads widen, liquidity dries up.
           * *1973 Oil Crisis*: Oil prices quadruple, stagflation spikes.
           * *2020 COVID Market Meltdown*: High-velocity panic drop, volatility index (VIX) exceeding 80.
           
        2. **Hypothetical Scenarios**: Modeling forward-looking plausibility:
           * *Geopolitical Conflict*: Disruption of trade routes leading to raw material price shock.
           * *Rate Tightening*: Aggressive rate hikes to combat stagflation, increasing discount rates.
           
        ### Factor Exposure & Volatility Spikes
        When a systemic crisis strikes, correlations between assets tend to **converge to 1.0** (diversification benefits disappear). Simultaneously, asset return distributions experience extreme **kurtosis (fat-tails)**. Stress testing allows risk managers to preemptively size positions to ensure they can survive these tail-risk regimes.
        """)
