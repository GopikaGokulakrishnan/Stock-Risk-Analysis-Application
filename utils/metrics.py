"""
Metrics Utility Module
----------------------
This module implements key statistical and financial metrics required for evaluation
of forecast predictions, return profiles, and risk-adjusted performance.

Formulas Included:
1. Root Mean Squared Error (RMSE)
2. Mean Absolute Error (MAE)
3. Mean Absolute Percentage Error (MAPE)
4. Sharpe Ratio
"""

import numpy as np
import pandas as pd
from typing import Union

def calculate_rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between actual and predicted values.
    
    Formula:
        RMSE = sqrt( (1 / N) * sum( (y_true - y_pred)^2 ) )
        
    Parameters:
        y_true (np.ndarray or pd.Series): Ground truth actual prices/values.
        y_pred (np.ndarray or pd.Series): Forecasted prices/values.
        
    Returns:
        float: The calculated RMSE score.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs y_true and y_pred must have the same length.")
    
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def calculate_mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Computes the Mean Absolute Error (MAE) between actual and predicted values.
    
    Formula:
        MAE = (1 / N) * sum( |y_true - y_pred| )
        
    Parameters:
        y_true (np.ndarray or pd.Series): Ground truth actual prices/values.
        y_pred (np.ndarray or pd.Series): Forecasted prices/values.
        
    Returns:
        float: The calculated MAE score.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs y_true and y_pred must have the same length.")
    
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_mape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Computes the Mean Absolute Percentage Error (MAPE) between actual and predicted values.
    
    Formula:
        MAPE = (100% / N) * sum( |(y_true - y_pred) / y_true| )
        
    Parameters:
        y_true (np.ndarray or pd.Series): Ground truth actual prices/values.
        y_pred (np.ndarray or pd.Series): Forecasted prices/values.
        
    Returns:
        float: The calculated MAPE score in percentage (%).
    """
    y_true, y_pred = np.array(y_true).astype(float), np.array(y_pred).astype(float)
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs y_true and y_pred must have the same length.")
    
    # Avoid division by zero by replacing zero actuals with a small epsilon
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    
    return float(np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100.0)


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    risk_free_rate: float = 0.07,
    periods_per_year: int = 252
) -> float:
    """
    Computes the Sharpe Ratio, measuring excess return per unit of volatility.
    
    Formula:
        Sharpe Ratio = (E[R_p] - R_f) / sigma_p
        Where:
            E[R_p] = Annualized mean return of the asset/portfolio.
            R_f    = Annualized risk-free rate of interest (e.g. 7% default for India).
            sigma_p = Annualized standard deviation of returns.
            
    Parameters:
        returns (Union[np.ndarray, pd.Series, list]): Periodic (daily) asset or portfolio returns.
        risk_free_rate (float): The annualized risk-free rate, defaulted to 0.07 (7.0%).
        periods_per_year (int): Frequency of observations, default 252 (trading days).
        
    Returns:
        float: The annualized Sharpe Ratio.
    """
    returns_arr = np.array(returns)
    if len(returns_arr) <= 1:
        return 0.0
    
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1)
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0
    
    # Annualize return and risk
    annualized_return = mean_return * periods_per_year
    annualized_volatility = std_return * np.sqrt(periods_per_year)
    
    excess_return = annualized_return - risk_free_rate
    return float(excess_return / annualized_volatility)
