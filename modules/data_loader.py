"""
Data Loader Module
------------------
This module fetches and pre-processes financial market data using Yahoo Finance (yfinance).
It leverages Streamlit caching (@st.cache_data) to prevent duplicate API requests and optimize speed.

It handles:
1. Historical price fetching (OHLCV)
2. Log returns and simple returns calculation
3. Missing value interpolation
4. Multiple ticker returns fetching for portfolio and correlation analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from typing import Tuple, Dict

_PRICE_COLS = ['Adj Close', 'Close']

def _select_price_column(df: pd.DataFrame) -> str:
    """Return a stable price column name from a downloaded yfinance dataframe."""
    for col in _PRICE_COLS:
        if col in df.columns:
            return col
    raise KeyError("No valid price column found. Expected one of: " + ", ".join(_PRICE_COLS))

@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent API throttling
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock market data for a single ticker.
    
    Parameters:
        ticker (str): The Yahoo Finance ticker symbol (e.g. 'RELIANCE.NS', 'TCS.NS', 'AAPL').
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: Stock price history containing Close, Open, High, Low, Volume, Adj Close,
                      along with computed Simple_Return and Log_Return columns.
    """
    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")
        
    try:
        # Fetch raw data with explicit pricing settings
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker} in the selected range.")
            
        # Clean multi-index columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Standardize missing data
        df = df.ffill().bfill()
        
        price_col = _select_price_column(df)
        if price_col != 'Adj Close':
            df = df.rename(columns={price_col: 'Adj Close'})
            price_col = 'Adj Close'
        
        # Calculate Returns
        # Simple Return = (P_t - P_t-1) / P_t-1
        df['Simple_Return'] = df[price_col].pct_change()
        
        # Log Return = ln(P_t / P_t-1)
        df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Drop rows where the selected price column is missing
        df = df.dropna(subset=[price_col])
        df = df.fillna(0) # clean remaining edge cases
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for ticker {ticker}: {str(e)}")
        raise e


@st.cache_data(ttl=3600)
def fetch_multiple_tickers(tickers: list, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches historical data for multiple tickers to build correlation matrices and
    portfolio portfolios.
    
    Parameters:
        tickers (list): List of ticker symbols (e.g. ['TCS.NS', 'INFY.NS', 'WIPRO.NS']).
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of Adjusted Close prices for all tickers
            - DataFrame of Log Returns for all tickers
    """
    if not tickers:
        raise ValueError("Ticker list cannot be empty.")
        
    prices = {}
    log_returns = {}
    
    for tk in tickers:
        try:
            df = yf.download(tk, start=start_date, end=end_date, auto_adjust=False, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                price_col = _select_price_column(df)
                prices[tk] = df[price_col].ffill().bfill()
                log_returns[tk] = np.log(prices[tk] / prices[tk].shift(1))
        except Exception as e:
            st.warning(f"Failed to fetch data for {tk}: {str(e)}")
            
    prices_df = pd.DataFrame(prices).dropna()
    returns_df = pd.DataFrame(log_returns).dropna()
    
    return prices_df, returns_df


def get_latest_financial_ratios(ticker: str) -> Dict[str, float]:
    """
    Fetches key financial ratios for a stock ticker from Yahoo Finance info.
    If yfinance limits access or returns empty/null, returns institutional defaults
    for robust calculation.
    
    Parameters:
        ticker (str): The ticker symbol.
        
    Returns:
        Dict[str, float]: Ratios dictionary containing Debt-to-Equity, Interest Coverage, etc.
    """
    # Institutional baseline defaults for safety
    defaults = {
        'debt_equity': 0.50,
        'interest_coverage': 4.5,
        'current_ratio': 1.8,
        'roe': 0.16,
        'net_profit_margin': 0.12,
        'operating_cash_flow': 50000000000.0, # in INR (e.g., 5000 Cr)
        'total_debt': 100000000000.0,
        'cash_and_equivalents': 20000000000.0,
        'shares_outstanding': 1000000000.0
    }
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or len(info) < 5:
            return defaults
            
        ratios = {
            'debt_equity': info.get('debtToEquity', defaults['debt_equity'] * 100.0) / 100.0, # convert % to ratio
            'interest_coverage': info.get('ebitda', 1.0) / max(info.get('totalDebt', 1.0) * 0.08, 1.0), # synthetic approx if missing
            'current_ratio': info.get('currentRatio', defaults['current_ratio']),
            'roe': info.get('returnOnEquity', defaults['roe']),
            'net_profit_margin': info.get('profitMargins', defaults['net_profit_margin']),
            'operating_cash_flow': info.get('operatingCashflow', defaults['operating_cash_flow']),
            'total_debt': info.get('totalDebt', defaults['total_debt']),
            'cash_and_equivalents': info.get('totalCash', defaults['cash_and_equivalents']),
            'shares_outstanding': info.get('sharesOutstanding', defaults['shares_outstanding'])
        }
        
        # Override specific approximate formulas to prevent divide-by-zero or negative results
        if ratios['interest_coverage'] <= 0:
            ratios['interest_coverage'] = defaults['interest_coverage']
        if ratios['debt_equity'] <= 0:
            ratios['debt_equity'] = defaults['debt_equity']
            
        return ratios
    except Exception:
        return defaults
