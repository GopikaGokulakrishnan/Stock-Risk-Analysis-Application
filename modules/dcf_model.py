"""
DCF Valuation Module
--------------------
This module implements a complete, interactive Discounted Cash Flow (DCF) model 
based on the Gordon Growth Formula.

Formulas:
1. Future Cash Flow Projection:
   FCF_t = FCF_0 * (1 + g)^t
2. Present Value of FCFs:
   PV(FCF_t) = FCF_t / (1 + WACC)^t
3. Terminal Value (TV):
   TV = FCF_n * (1 + g_terminal) / (WACC - g_terminal)
4. Present Value of Terminal Value:
   PV(TV) = TV / (1 + WACC)^n
5. Enterprise Value (EV):
   EV = Sum(PV(FCF_t)) + PV(TV)
6. Equity Value:
   Equity Value = EV + Cash - Debt
7. Intrinsic Value per share:
   Intrinsic Value = Equity Value / Shares Outstanding
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import get_plotly_layout

def run_dcf_module(
    ticker: str,
    current_price: float,
    financial_ratios: dict,
    wacc: float = 0.10,
    terminal_growth: float = 0.05,
    fcf_growth: float = 0.08,
    projection_years: int = 5
) -> float:
    """
    Runs the DCF model, renders a waterfall chart of enterprise value creation,
    displays intrinsic value & margin of safety metrics, and returns the intrinsic value per share.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        current_price (float): The current market price.
        financial_ratios (dict): Dictionary containing total debt, cash, and shares outstanding.
        wacc (float): Weighted Average Cost of Capital (ratio, e.g. 0.10 for 10%).
        terminal_growth (float): Terminal Growth Rate (ratio, e.g. 0.05 for 5%).
        fcf_growth (float): Forecasted annual growth of cash flows (ratio, e.g. 0.08 for 8%).
        projection_years (int): Number of years to project, default 5.
        
    Returns:
        float: Calculated Intrinsic Value per share.
    """
    st.markdown(f"### 🏢 Discounted Cash Flow (DCF) Valuation: {ticker}")
    
    # Check WACC > Terminal Growth to prevent Gordon Growth divide-by-zero or negative terminal value
    if wacc <= terminal_growth:
        st.warning("⚠️ WACC must be strictly greater than the Terminal Growth Rate for the Gordon Growth Model to converge. Resetting WACC to Growth + 2%.")
        wacc = terminal_growth + 0.02
        
    # Extract structural metrics
    fcf_base = financial_ratios['operating_cash_flow']
    total_debt = financial_ratios['total_debt']
    cash_equiv = financial_ratios['cash_and_equivalents']
    shares = financial_ratios['shares_outstanding']
    
    # 1. Cash Flow Projections
    years = list(range(1, projection_years + 1))
    projected_fcf = []
    pv_projected_fcf = []
    
    current_fcf = fcf_base
    for yr in years:
        # Project
        current_fcf = current_fcf * (1 + fcf_growth)
        projected_fcf.append(current_fcf)
        # Discount
        pv = current_fcf / ((1 + wacc) ** yr)
        pv_projected_fcf.append(pv)
        
    sum_pv_fcf = sum(pv_projected_fcf)
    
    # 2. Terminal Value Calculation (Gordon Growth Model)
    # TV = FCF_n * (1 + g_term) / (WACC - g_term)
    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
    
    # 3. Enterprise & Equity Valuation
    enterprise_value = sum_pv_fcf + pv_terminal_value
    net_debt = total_debt - cash_equiv
    equity_value = enterprise_value - net_debt
    
    # Intrinsic value per share
    intrinsic_value = equity_value / shares
    
    if intrinsic_value < 0:
        intrinsic_value = 0.0
        
    # Margin of Safety
    # MOS = (Intrinsic Value - Current Price) / Intrinsic Value
    if intrinsic_value > 0:
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
    else:
        margin_of_safety = 0.0
        
    # Valuation Status
    if margin_of_safety > 0.20:
        status = "UNDERVALUED 🟢"
        status_color = "#10B981"
    elif -0.10 <= margin_of_safety <= 0.20:
        status = "FAIRLY VALUED 🟡"
        status_color = "#F59E0B"
    else:
        status = "OVERVALUED 🔴"
        status_color = "#EF4444"

    # Display Metrics Grid
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("DCF Intrinsic Value", f"₹ {intrinsic_value:,.2f}", 
                  delta=f"{margin_of_safety*100:.2f}% Margin")
    with d_col2:
        st.metric("Valuation Status", status, help="Based on discount margins")
    with d_col3:
        st.metric("Enterprise Value (EV)", f"₹ {enterprise_value:,.0f}", 
                  help="Total operating value of the firm")
    with d_col4:
        st.metric("Net Debt Structure", f"₹ {net_debt:,.0f}", 
                  help="Total Debt minus Cash & Equivalents")
        
    # Secondary Data Inputs Used Info
    with st.expander("🛠️ View Inputs & Cash Flow Schedule"):
        st.markdown(f"""
        * **Base Operating Cash Flow (FCF_0)**: ₹ {fcf_base:,.0f}
        * **Debt / Cash**: Debt ₹ {total_debt:,.0f} | Cash ₹ {cash_equiv:,.0f}
        * **Shares Outstanding**: {shares:,.0f}
        """)
        # Display schedule as table
        schedule_df = pd.DataFrame({
            "Projected FCF (₹)": [f"₹ {v:,.0f}" for v in projected_fcf],
            "PV Factor": [f"{(1 / (1 + wacc)**y):.4f}" for y in years],
            "Discounted PV (₹)": [f"₹ {v:,.0f}" for v in pv_projected_fcf]
        }, index=[f"Year {y}" for y in years])
        st.table(schedule_df)

    # 4. Plotly Waterfall Chart (Enterprise Value breakdown to Equity Value)
    fig = go.Figure()
    
    fig.add_trace(
        go.Waterfall(
            name="DCF Waterfall",
            orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["PV of Cash Flows", "PV of Terminal Value", "Enterprise Value", "Net Debt Subtraction", "Net Equity Value"],
            textposition="outside",
            text=[
                f"₹ {sum_pv_fcf/1e9:.2f}B",
                f"₹ {pv_terminal_value/1e9:.2f}B",
                f"₹ {enterprise_value/1e9:.2f}B",
                f"-₹ {net_debt/1e9:.2f}B",
                f"₹ {equity_value/1e9:.2f}B"
            ],
            y=[sum_pv_fcf, pv_terminal_value, enterprise_value, -net_debt, equity_value],
            connector={"line": {"color": "#475569", "width": 1.5, "dash": "dot"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#06B6D4"}}
        )
    )
    
    # Apply layout custom theme
    waterfall_layout = get_plotly_layout("Discounted Cash Flow (DCF) Capital Structure Breakdown", "", "Valuation Value (₹)")
    fig.update_layout(waterfall_layout)
    st.plotly_chart(fig, use_container_width=True)
    
    # Financial Theory Quick Reference (Viva prep)
    with st.expander("📚 Theoretical Framework: DCF & Gordon Growth Model"):
        st.markdown(r"""
        ### The Discounted Cash Flow (DCF) Concept
        The central premise of DCF is that the value of an asset is equal to the sum of its future cash flows, discounted back to the present day using an appropriate discount rate. The discount rate represents the **opportunity cost** of capital and the riskiness of the cash flows.
        
        ### Weighted Average Cost of Capital (WACC)
        WACC is the combined cost of debt and equity financing. It is the hurdle rate that a firm must beat to create value for investors:
        $$\text{WACC} = \left(\frac{E}{V} \times R_e\right) + \left(\frac{D}{V} \times R_d \times (1 - T_c)\right)$$
        where $E$ is Equity, $D$ is Debt, $V = E+D$, $R_e$ is Cost of Equity (CAPM), $R_d$ is Cost of Debt, and $T_c$ is Corporate Tax.
        
        ### Terminal Value (TV) & Gordon Growth Model
        A business is assumed to be a going concern that lives indefinitely. Since we cannot forecast cash flows for 100 years, we project cash flows for a specific period (e.g. 5 years) and calculate a **Terminal Value** to capture the value of all cash flows beyond year 5.
        
        The **Gordon Growth Model** calculates this by assuming cash flows grow at a constant rate $g_{terminal}$ forever:
        $$\text{TV} = \frac{\text{FCF}_n \times (1 + g_{\text{terminal}})}{\text{WACC} - g_{\text{terminal}}}$$
        
        * **Constraint**: WACC *must* be strictly greater than $g_{terminal}$ for this formula to remain positive and mathematically valid.
        
        ### Margin of Safety
        Calculated as the premium of intrinsic value over market price. A 20%+ margin of safety is standard in institutional value investing to provide protection against analytical error or unexpected market distress.
        """)
        
    return float(intrinsic_value)
