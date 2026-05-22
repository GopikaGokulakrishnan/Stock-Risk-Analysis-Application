"""
Value at Risk (VaR) Module
--------------------------
This module implements market risk measures: Value at Risk (VaR) and Conditional 
Value at Risk (CVaR, Expected Shortfall) across three core methodologies:
1. Historical Simulation VaR
2. Parametric (Variance-Covariance) VaR
3. Monte Carlo VaR

Additionally, it integrates a Kupiec Proportion of Failures (POF) Likelihood Ratio
Backtesting system to statistically validate the risk estimates.

Theoretical Concepts:
- VaR: Maximum loss expected over a horizon at a confidence level.
- CVaR: Average loss in the worst-case tail beyond the VaR threshold.
- Kupiec Test: A Likelihood Ratio (LR) test verifying if the number of actual 
  losses exceeding VaR is statistically consistent with the expected rate (1 - alpha).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, chi2
from utils.helpers import get_plotly_layout

def run_var_module(ticker: str, df: pd.DataFrame, confidence_level: float = 0.95) -> float:
    """
    Computes Historical, Parametric, and Monte Carlo VaR & CVaR, performs Kupiec
    backtesting, visualizes the loss distribution, and returns the historical 95% VaR.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        df (pd.DataFrame): Stock historical price dataframe.
        confidence_level (float): Confidence level, default 0.95 (95%).
        
    Returns:
        float: Calculated historical VaR (as a daily ratio e.g. -0.024 for -2.4% daily risk).
    """
    st.markdown(f"### 🛡️ Market Risk: Value at Risk (VaR) & CVaR: {ticker}")
    
    returns = df['Log_Return'].dropna()
    
    if len(returns) < 100:
        st.warning("Insufficient return history to compute robust VaR models.")
        return -0.02
        
    # Standard parameters
    alpha = 1 - confidence_level # Significance level e.g. 0.05
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    # 1. Historical VaR & CVaR
    # VaR is the alpha-percentile of returns
    var_hist = np.percentile(returns, alpha * 100)
    cvar_hist = returns[returns <= var_hist].mean()
    
    # 2. Parametric (Normal) VaR & CVaR
    # VaR = mean - Z_alpha * std
    z_score = norm.ppf(confidence_level)
    var_param = mean_ret - z_score * std_ret
    
    # CVaR parametric formula under normality:
    # CVaR = mean - std * (phi(Z) / (1 - alpha))
    # where phi is standard normal PDF
    cvar_param = mean_ret - std_ret * (norm.pdf(z_score) / alpha)
    
    # 3. Monte Carlo VaR & CVaR
    # Simulate 10,000 returns using the fitted normal distribution
    np.random.seed(42)
    mc_returns = np.random.normal(mean_ret, std_ret, 10000)
    var_mc = np.percentile(mc_returns, alpha * 100)
    cvar_mc = mc_returns[mc_returns <= var_mc].mean()
    
    # Convert to percentage representation for UI display
    var_hist_pct = var_hist * 100.0
    cvar_hist_pct = cvar_hist * 100.0
    var_param_pct = var_param * 100.0
    cvar_param_pct = cvar_param * 100.0
    var_mc_pct = var_mc * 100.0
    cvar_mc_pct = cvar_mc * 100.0
    
    # 4. Kupiec POF Backtesting
    # We count how many historical returns were worse than (below) the VaR threshold
    failures = np.sum(returns < var_hist)
    total_obs = len(returns)
    failure_rate = failures / total_obs
    
    # Kupiec Likelihood Ratio (LR) test statistic
    # Null Hypothesis (H0): Model's failure rate equals alpha (expected rate)
    # LR = -2 * ln( ((1-alpha)^(T-N) * alpha^N) / ((1-pi)^(T-N) * pi^N) )
    # where T = total_obs, N = failures, pi = failure_rate
    # If N = 0 or failure_rate = 1, we handle mathematically
    if failures > 0 and failure_rate < 1:
        ln_num = (total_obs - failures) * np.log(1 - alpha) + failures * np.log(alpha)
        ln_den = (total_obs - failures) * np.log(1 - failure_rate) + failures * np.log(failure_rate)
        lr_stat = -2 * (ln_num - ln_den)
    else:
        lr_stat = 0.0
        
    # Critical value of Chi-square with 1 degree of freedom at 95% confidence is 3.84
    p_value = 1.0 - chi2.cdf(lr_stat, df=1) if lr_stat > 0 else 1.0
    model_valid = "VALID / ACCEPTABLE ✅" if lr_stat <= 3.84 else "REJECTED / UNRELIABLE ❌"
    model_valid_color = "#10B981" if lr_stat <= 3.84 else "#EF4444"

    # Display Metrics Grid
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.markdown(f"**Historical Simulation**")
        st.metric("1-Day VaR (95%)", f"{var_hist_pct:.2f}%")
        st.metric("Expected Shortfall (CVaR)", f"{cvar_hist_pct:.2f}%")
    with v_col2:
        st.markdown(f"**Parametric (Gaussian)**")
        st.metric("1-Day VaR (95%)", f"{var_param_pct:.2f}%")
        st.metric("Expected Shortfall (CVaR)", f"{cvar_param_pct:.2f}%")
    with v_col3:
        st.markdown(f"**Monte Carlo (10K Paths)**")
        st.metric("1-Day VaR (95%)", f"{var_mc_pct:.2f}%")
        st.metric("Expected Shortfall (CVaR)", f"{cvar_mc_pct:.2f}%")
        
    st.markdown("---")
    
    # Kupiec backtest panel
    st.markdown("#### 🧪 Kupiec Proportion of Failures Backtest Summary")
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        st.metric("Historical Violations", f"{failures} days", help=f"Expected violations: {total_obs * alpha:.1f} days")
    with k_col2:
        st.metric("Observed Failure Rate", f"{failure_rate*100:.2f}%", delta=f"{(failure_rate - alpha)*100:.2f}% vs expected")
    with k_col3:
        st.metric("Kupiec LR Statistic", f"{lr_stat:.4f}", help="Chi-Square (df=1) test. Critical value is 3.841")
    with k_col4:
        st.markdown(f"<div style='padding-top: 10px;'><span style='color: {model_valid_color}; font-weight: 700; font-size: 1.1rem;'>{model_valid}</span><br/><span style='font-size:0.75rem; color:#64748B;'>LR P-Value: {p_value:.4f}</span></div>", unsafe_allow_html=True)
        
    st.markdown("---")

    # 5. Plotly Return Distribution Chart showing VaR and CVaR
    fig = go.Figure()
    
    # Plot returns histogram
    fig.add_trace(
        go.Histogram(
            x=returns * 100.0, # Convert to %
            nbinsx=50,
            name='Daily Log Returns (%)',
            marker_color='#06B6D4',
            opacity=0.6,
            histnorm='probability density'
        )
    )
    
    # Fit normal curve for visual reference
    x_range = np.linspace(returns.min(), returns.max(), 200)
    y_normal = norm.pdf(x_range, mean_ret, std_ret)
    fig.add_trace(
        go.Scatter(
            x=x_range * 100.0,
            y=y_normal / 100.0, # Scaled density
            mode='lines',
            name='Gaussian Normal Fit',
            line=dict(color='#64748B', width=2)
        )
    )
    
    # Add VaR & CVaR thresholds (Historical)
    fig.add_vline(x=var_hist_pct, line_width=3, line_color="#F59E0B", annotation_text=f"Historical VaR: {var_hist_pct:.2f}%")
    fig.add_vline(x=cvar_hist_pct, line_width=3, line_color="#EF4444", annotation_text=f"Historical CVaR: {cvar_hist_pct:.2f}%")
    
    fig.update_layout(get_plotly_layout(f"Daily Asset Return Distribution & Downside Tail Risk", "Daily Returns (%)", "Density"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Theoretical Explanation
    with st.expander("📚 Theoretical Framework: Value at Risk (VaR) & Kupiec Backtesting"):
        st.markdown(r"""
        ### Value at Risk (VaR) Concept
        Value at Risk (VaR) represents the maximum potential loss over a specific target horizon (e.g. 1 day) within a given confidence level $\alpha$ (e.g. 95%), assuming normal market behavior. It answers the question: *"What is the maximum percentage I could lose tomorrow with 95% confidence?"*
        
        ### Three VaR Methodologies:
        1. **Historical Simulation (Non-parametric)**: 
           Does not assume a specific probability distribution. It takes all actual historical daily returns, orders them from worst to best, and locates the $(1-\text{confidence})$-th percentile return.
           * **Pros**: Captures real market properties (fat-tails, skewness, historical anomalies).
           * **Cons**: Assumes history will repeat itself.
           
        2. **Parametric / Variance-Covariance (Gaussian)**:
           Assumes returns are normally distributed. It calculates VaR using the mean ($\mu$) and standard deviation ($\sigma$):
           $$\text{VaR} = \mu - Z_{\alpha} \sigma$$
           where $Z_{\alpha}$ is the standard normal z-score (e.g. $1.645$ for 95%).
           * **Pros**: Simple, mathematically elegant, fast.
           * **Cons**: Underestimates tail risks because actual financial returns have "fat tails" (kurtosis > 3).
           
        3. **Monte Carlo Simulation**:
           Generates thousands of random price scenarios based on a model (like Geometric Brownian Motion) and identifies the percentile loss from these simulated values.
           * **Pros**: Highly flexible; can incorporate complex derivative structures or changing conditions.
           * **Cons**: Computationally heavy, prone to model specification risk.
           
        ### What is CVaR / Expected Shortfall?
        While VaR tells us the *threshold* of losses, it does not tell us *how bad* the loss will be if we cross that threshold. **Conditional Value at Risk (CVaR)**, or Expected Shortfall, calculates the average return in the worst-case tail:
        $$\text{CVaR}_{\alpha} = E[R \mid R \le \text{VaR}_{\alpha}]$$
        CVaR is a **coherent risk measure** because it satisfies subadditivity (the risk of a combined portfolio is always less than or equal to the sum of individual risks).
        
        ### Kupiec Likelihood Ratio (LR) Test
        The **Kupiec Proportion of Failures (POF)** test is used to validate risk models. A model is accurate if the failure rate $\pi = N/T$ is statistically close to the expected failure rate $p = 1 - \alpha$.
        
        The Likelihood Ratio test statistic is:
        $$LR = -2 \ln \left( \frac{(1 - p)^{T-N} p^N}{(1 - \pi)^{T-N} \pi^N} \right)$$
        This follows a Chi-square distribution with 1 degree of freedom: $LR \sim \chi^2(1)$.
        * **Evaluation**: If $LR > 3.841$ (at the 95% confidence level), we **reject** the model as invalid, indicating the VaR model has poor calibration.
        """)
        
    return float(var_hist)
