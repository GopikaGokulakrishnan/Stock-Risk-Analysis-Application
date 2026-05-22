"""
GARCH Volatility Module
-----------------------
This module implements volatility modeling using the ARCH library (GARCH(1,1) model).
It extracts conditional volatility, compares it to rolling historical volatility,
detects high-volatility regimes, and visualizes volatility spikes with Plotly annotations.

Mathematical Model:
    sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2
    
Theoretical Concepts:
1. Volatility Clustering: High-volatility periods group together.
2. Heteroskedasticity: Time-varying variance.
3. Volatility vs Returns: Returns measure price change direction; volatility measures rate/dispersion.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model
from utils.helpers import get_plotly_layout

def run_garch_module(ticker: str, df: pd.DataFrame) -> float:
    """
    Fits a GARCH(1,1) model to stock returns, detects regimes, visualizes
    volatility metrics, and returns the latest annualized conditional volatility.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        df (pd.DataFrame): Stock historical dataframe.
        
    Returns:
        float: Latest annualized GARCH conditional volatility (ratio e.g. 0.22 for 22%).
    """
    st.markdown(f"### ⚡ Volatility Modeling: GARCH(1,1) Engine: {ticker}")
    
    returns = df['Log_Return'].dropna()
    
    if len(returns) < 100:
        st.warning("Insufficient data for fitting a robust GARCH volatility model.")
        return 0.20
        
    # Scale returns by 100 to ensure GARCH optimizer convergence
    scaled_returns = returns * 100.0
    
    # 1. Fit GARCH(1,1) Model
    with st.spinner("Fitting GARCH(1,1) conditional volatility model..."):
        try:
            model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal', rescale=False)
            res = model.fit(disp='off')
            
            # Extract daily conditional volatility (divide by 100 to unscale)
            cond_vol_daily = res.conditional_volatility / 100.0
            
            # Annualize conditional volatility
            cond_vol_ann = cond_vol_daily * np.sqrt(252)
            
            # Extract GARCH coefficients
            omega = res.params['omega'] / 100.0 # Unscale
            alpha = res.params['alpha[1]']
            beta = res.params['beta[1]']
            persistence = alpha + beta
            
        except Exception as e:
            st.warning(f"GARCH optimization failed to converge: {str(e)}. Falling back to rolling historical estimate.")
            # Fallback to rolling volatility
            cond_vol_daily = returns.rolling(window=20).std().fillna(returns.std())
            cond_vol_ann = cond_vol_daily * np.sqrt(252)
            omega, alpha, beta, persistence = 0.0001, 0.05, 0.90, 0.95
            
    # 2. Compute Rolling Volatility (20-day historical window) for comparison
    rolling_vol_daily = returns.rolling(window=20).std()
    rolling_vol_ann = rolling_vol_daily * np.sqrt(252)
    
    # Latest Volatility Metrics
    latest_garch_vol = cond_vol_ann.iloc[-1]
    latest_rolling_vol = rolling_vol_ann.iloc[-1]
    
    # 3. Volatility Regime Detection
    # Volatility Spike defined as exceeding the 90th percentile of conditional volatility
    threshold = cond_vol_ann.quantile(0.90)
    spikes = cond_vol_ann[cond_vol_ann > threshold]
    
    # Detect the most recent spike date
    latest_spike_str = "No recent spike"
    if not spikes.empty:
        latest_spike_date = spikes.index[-1]
        latest_spike_str = f"{latest_spike_date.strftime('%Y-%m-%d')} ({spikes.iloc[-1]*100:.2f}%)"

    # Display Metrics Grid
    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    with c_col1:
        st.metric("GARCH Volatility (Annual)", f"{latest_garch_vol*100:.2f}%", 
                  delta=f"{(latest_garch_vol - latest_rolling_vol)*100:.2f}% vs Hist")
    with c_col2:
        st.metric("Model Persistence (α + β)", f"{persistence:.3f}", 
                  help="Measure of shock absorption speed. Closer to 1.0 means shocks linger longer.")
    with c_col3:
        st.metric("α (Reaction) / β (Decay)", f"{alpha:.3f} / {beta:.3f}",
                  help="Alpha measures short-term shock response; Beta measures long-term variance dependency.")
    with c_col4:
        st.metric("Spike Threshold (90th Pctl)", f"{threshold*100:.2f}%", 
                  help=f"Latest Vol Spike: {latest_spike_str}")

    # 4. Plotly Chart
    fig = go.Figure()
    
    # Plot GARCH Conditional Volatility
    fig.add_trace(
        go.Scatter(
            x=cond_vol_ann.index,
            y=cond_vol_ann.values,
            mode='lines',
            name='GARCH(1,1) Conditional Volatility',
            line=dict(color='#F43F5E', width=2) # Bright Red/Rose
        )
    )
    
    # Plot Historical Rolling Volatility
    fig.add_trace(
        go.Scatter(
            x=rolling_vol_ann.index,
            y=rolling_vol_ann.values,
            mode='lines',
            name='20-Day Rolling Historical Volatility',
            line=dict(color='#64748B', width=1.5, dash='dot') # Gray dashed
        )
    )
    
    # Annotate Volatility Spikes
    # Add scatter markers on spike points
    if not spikes.empty:
        # Downsample spikes for visual clarity
        spikes_downsampled = spikes.resample('ME').max().dropna()
        fig.add_trace(
            go.Scatter(
                x=spikes_downsampled.index,
                y=spikes_downsampled.values,
                mode='markers',
                name='Vol Regime Shifts / Spikes',
                marker=dict(color='#EF4444', size=8, symbol='triangle-up'),
                hoverinfo='text',
                hovertext=[f"Spike: {val*100:.2f}%" for val in spikes_downsampled.values]
            )
        )
        
    fig.update_layout(get_plotly_layout(f"GARCH(1,1) vs. Historical Volatility (Annualized)", "Timeline", "Volatility (%)"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Theoretical Explanation
    with st.expander("📚 Theoretical Framework: Volatility Clustering & GARCH"):
        st.markdown(r"""
        ### Why GARCH is Used in Quantitative Finance
        Unlike traditional statistical models which assume **homoskedasticity** (constant variance), asset returns exhibit **heteroskedasticity** (variance that changes over time). GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) was developed by Robert Engle (Nobel laureate) to capture this time-varying risk. It recognizes that volatility is not constant and can be modeled dynamically.
        
        ### GARCH(1,1) Mathematical Formula:
        The conditional variance $\sigma_t^2$ at time $t$ is expressed as:
        $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
        
        Where:
        * **$\sigma_t^2$**: Conditional variance for today (variance given historical information).
        * **$\omega$**: Baseline long-term variance constant.
        * **$\alpha$ (ARCH parameter)**: Coefficient on the lagged squared residual ($\epsilon_{t-1}^2$). It measures the **short-term response** to market shocks (news).
        * **$\beta$ (GARCH parameter)**: Coefficient on the lagged conditional variance ($\sigma_{t-1}^2$). It measures the **long-term decay** or persistence of volatility.
        * **$\alpha + \beta$ (Persistence)**: Volatility persistence. If $\alpha + \beta \approx 1$, volatility shocks decay very slowly (volatility clustering). If $\alpha + \beta \ge 1$, the process is non-stationary (unbounded volatility).
        
        ### Difference Between Volatility and Returns
        * **Returns**: Measures the *direction* and magnitude of price changes ($P_t - P_{t-1}$).
        * **Volatility**: Measures the *dispersion* of returns (uncertainty/risk) without regard to direction. An asset can have zero return over a month, but experience extreme volatility during that month.
        
        ### Volatility Clustering
        First noted by Benoit Mandelbrot: *"Large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes."* This is visible in GARCH charts as clustering of volatility waves during periods of market stress (recessions, geopolitical shifts).
        """)
        
    return float(latest_garch_vol)
