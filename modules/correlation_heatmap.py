"""
Correlation Heatmap Module
--------------------------
This module implements asset correlation modeling for a portfolio basket.
It computes the Pearson Correlation Matrix of selected assets, renders a beautifully
annotated Plotly Heatmap, and extracts strategic diversification insights.

Theoretical Concepts:
- Correlation Coefficient (r): Pearson product-moment coefficient bounded between -1.0 and +1.0.
- Positive vs Negative Correlation: Positive means assets move together; negative means opposite moves.
- Diversification Benefit: Holding uncorrelated assets to lower overall portfolio standard deviation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import get_plotly_layout
from modules.data_loader import fetch_multiple_tickers

def run_correlation_module(
    selected_tickers: list,
    start_date: str,
    end_date: str
) -> None:
    """
    Computes multilateral correlations, plots the annotated Plotly heatmap,
    extracts diversification pairs, and provides quantitative insights.
    
    Parameters:
        selected_tickers (list): List of ticker symbols to correlate.
        start_date (str): Fetch start date.
        end_date (str): Fetch end date.
    """
    st.markdown(f"### 🔀 Asset Correlation Heatmap & Diversification Engine")
    
    if len(selected_tickers) < 2:
        st.warning("⚠️ Please select at least 2 tickers in the sidebar toggles to build a correlation heatmap.")
        return
        
    # 1. Fetch Multilateral Data
    with st.spinner("Fetching pricing history and calculating log returns..."):
        prices_df, returns_df = fetch_multiple_tickers(selected_tickers, start_date, end_date)
        
    if returns_df.empty or len(returns_df.columns) < 2:
        st.error("Failed to load sufficient multilateral pricing history.")
        return
        
    # 2. Compute Pearson Correlation Matrix
    corr_matrix = returns_df.corr(method='pearson')
    
    # Round for display
    corr_matrix_round = corr_matrix.round(3)
    
    # 3. Plotly Annotated Heatmap
    z_values = corr_matrix_round.values
    x_labels = list(corr_matrix_round.columns)
    y_labels = list(corr_matrix_round.index)
    
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='RdBu_r', # Red is highly positive, Blue is highly negative (flipped for finance)
            zmin=-1.0,
            zmax=1.0,
            showscale=True,
            colorbar=dict(title='Pearson Correlation', thickness=15)
        )
    )
    
    # Add Heatmap cell text annotations
    annotations = []
    for y_idx in range(len(y_labels)):
        for x_idx in range(len(x_labels)):
            val = z_values[y_idx][x_idx]
            # determine text color based on contrast
            text_color = "#FFFFFF" if abs(val) > 0.4 else "#1E293B"
            annotations.append(
                dict(
                    x=x_labels[x_idx],
                    y=y_labels[y_idx],
                    text=f"{val:.3f}",
                    font=dict(family="JetBrains Mono, monospace", size=13, color=text_color),
                    showarrow=False
                )
            )
            
    fig.update_layout(annotations=annotations)
    fig.update_layout(get_plotly_layout("Pearson Correlation Matrix", "", ""))
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. Extract Strategic Diversification Insights
    # Find most and least correlated asset pairs
    corr_unstacked = corr_matrix.unstack()
    # Filter out identity correlations (diagonal self-correlations of 1.0)
    corr_unstacked_filtered = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
    
    if not corr_unstacked_filtered.empty:
        # Find absolute max and min
        max_pair = corr_unstacked_filtered.idxmax()
        max_val = corr_unstacked_filtered.max()
        
        min_pair = corr_unstacked_filtered.idxmin()
        min_val = corr_unstacked_filtered.min()
        
        # Display insights
        st.markdown("#### 🔍 Portfolio Diversification Insights")
        
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            st.info(f"""
            📈 **Highest Correlation Pair**:
            * **Assets**: `{max_pair[0]}` & `{max_pair[1]}`
            * **Pearson r**: `{max_val:.4f}`
            * **Strategic Risk**: These assets move in tight synchronization. Holding both provides minimal diversification. Risk is concentrated.
            """)
        with col_in2:
            st.success(f"""
            📉 **Lowest Correlation Pair**:
            * **Assets**: `{min_pair[0]}` & `{min_pair[1]}`
            * **Pearson r**: `{min_val:.4f}`
            * **Strategic Benefit**: This pair provides the **strongest diversification benefit**. Adding them together effectively suppresses overall portfolio variance!
            """)
            
        # Overall Portfolio Diversification Grade
        mean_corr = corr_unstacked_filtered.mean()
        if mean_corr > 0.70:
            grade = "POOR (Highly Concentrated) 🔴"
            grade_detail = "Average cross-asset correlation is very high. Your portfolio is highly exposed to sector or market-wide systemic shocks."
        elif 0.35 <= mean_corr <= 0.70:
            grade = "MODERATE (Standard Diversification) 🟡"
            grade_detail = "Average correlation is in the medium range. Typical of large-cap blue-chip equities in related sectors."
        else:
            grade = "EXCELLENT (Strongly Diversified) 🟢"
            grade_detail = "Average correlation is very low. Portfolio components provide excellent risk offsets, dampening downside volatility."
            
        st.markdown(f"""
        * **Portfolio Diversification Quality**: **{grade}**
        * **Evaluation**: {grade_detail} (Average Correlation: `{mean_corr:.4f}`)
        """)
        
    # Theoretical Explanation
    with st.expander("📚 Theoretical Framework: Correlation & Diversification"):
        st.markdown(r"""
        ### Pearson Correlation Coefficient Formula
        Pearson correlation ($r$) measures the strength and direction of the linear relationship between two continuous variables. For two asset returns $X$ and $Y$:
        $$r_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$$
        
        Where:
        * **$\text{Cov}(X,Y)$**: Covariance of returns (how they vary together).
        * **$\sigma_X, \sigma_Y$**: Individual standard deviations (individual asset risks).
        
        ### Bounds and Interpretation:
        * **$r = +1.0$ (Perfect Positive Correlation)**: The assets move in perfect lockstep. No diversification benefit.
        * **$r = 0.0$ (Uncorrelated)**: Asset movements are independent. Excellent diversification.
        * **$r = -1.0$ (Perfect Negative Correlation)**: Assets move in opposite directions. Perfect hedge (but zero expected returns if equal and opposite).
        
        ### The Diversification Benefit
        The variance of a 2-asset portfolio is:
        $$\sigma_p^2 = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 \sigma_{12} = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 r_{12} \sigma_1 \sigma_2$$
        
        * Notice that the third term is scaled by the correlation coefficient $r_{12}$.
        * If $r_{12} < 1.0$, the portfolio standard deviation $\sigma_p$ is strictly *less* than the weighted average of individual standard deviations ($w_1 \sigma_1 + w_2 \sigma_2$). This reduction in standard deviation with no loss in expected return is the **"only free lunch in finance."**
        """)
