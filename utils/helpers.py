"""
Visual & UI Helper Module
--------------------------
This module provides standardized design utilities to inject custom Bloomberg-style
CSS into the Streamlit app and apply professional, consistent styling to Plotly charts.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

def inject_bloomberg_css() -> None:
    """
    Injects custom CSS to style the Streamlit interface to look like an institutional
    Bloomberg Terminal / TradingView platform. Sets high-contrast dark themes, sleek
    fonts, customized sidebar borders, and modern card tiles.
    """
    bloomberg_theme = """
    <style>
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Apply Base Font Styles */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #0B0E14 !important;
            color: #F8FAFC !important;
        }
        
        /* Main Container Padding */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 95% !important;
        }

        /* Customize Streamlit Sidebar */
        [data-testid="stSidebar"] {
            background-color: #11151F !important;
            border-right: 1px solid #1E293B !important;
        }
        
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem !important;
        }

        /* Custom KPI Card Styling */
        .kpi-card {
            background: linear-gradient(135deg, #11151F 0%, #1E2535 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: #06B6D4; /* Bright Cyan border on hover */
        }
        
        .kpi-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #94A3B8;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #F8FAFC;
            margin-bottom: 0.25rem;
        }
        
        .kpi-subtext {
            font-size: 0.75rem;
            color: #64748B;
        }
        
        /* Status Badges */
        .badge-buy {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .badge-hold {
            background-color: rgba(100, 116, 139, 0.15);
            color: #94A3B8;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
            border: 1px solid rgba(100, 116, 139, 0.3);
        }
        
        .badge-sell {
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Custom Header Styling */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }

        /* Terminal block */
        .terminal-block {
            font-family: 'JetBrains Mono', monospace;
            background-color: #05070B !important;
            border: 1px solid #1E293B;
            border-left: 4px solid #F59E0B;
            padding: 1rem;
            border-radius: 6px;
            color: #10B981 !important;
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }
    </style>
    """
    st.markdown(bloomberg_theme, unsafe_allow_html=True)


def get_plotly_layout(title: str, x_title: str = "", y_title: str = "") -> Dict[str, Any]:
    """
    Generates a consistent institutional Plotly chart layout template (Bloomberg Theme).
    
    Parameters:
        title (str): Title of the chart.
        x_title (str): Title of the X-axis (optional).
        y_title (str): Title of the Y-axis (optional).
        
    Returns:
        Dict[str, Any]: A Plotly layout configuration dictionary.
    """
    return {
        "title": {
            "text": title,
            "font": {"family": "Outfit, sans-serif", "size": 18, "color": "#F8FAFC"},
            "x": 0.05
        },
        "paper_bgcolor": "#11151F",
        "plot_bgcolor": "#0B0E14",
        "font": {"family": "Outfit, sans-serif", "color": "#94A3B8"},
        "xaxis": {
            "title": x_title,
            "gridcolor": "#1E293B",
            "zerolinecolor": "#334155",
            "showgrid": True,
            "showline": True,
            "linecolor": "#334155"
        },
        "yaxis": {
            "title": y_title,
            "gridcolor": "#1E293B",
            "zerolinecolor": "#334155",
            "showgrid": True,
            "showline": True,
            "linecolor": "#334155"
        },
        "margin": {"l": 50, "r": 30, "t": 60, "b": 50},
        "legend": {
            "bgcolor": "#11151F",
            "bordercolor": "#334155",
            "borderwidth": 1,
            "font": {"size": 11}
        },
        "hovermode": "x unified",
        "autosize": True
    }


def create_kpi_card(title: str, value: str, subtext: str = "", css_class: str = "") -> str:
    """
    Helper function to generate beautiful HTML code for custom KPI cards.
    
    Parameters:
        title (str): Metric name/label.
        value (str): Main metric value.
        subtext (str): Supporting commentary or percentage change.
        css_class (str): Additional CSS styles or badges.
        
    Returns:
        str: Raw HTML for Streamlit markdown.
    """
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">{subtext} {css_class}</div>
    </div>
    """
