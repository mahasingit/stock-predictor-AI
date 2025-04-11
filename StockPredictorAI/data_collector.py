import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical stock/crypto data from Yahoo Finance.
    
    Parameters:
    ticker (str): Symbol of the stock/cryptocurrency
    start_date (datetime): Start date for data retrieval
    end_date (datetime): End date for data retrieval
    
    Returns:
    pandas.DataFrame: DataFrame containing the historical data
    """
    try:
        # Convert dates to string format for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_str, end=end_str)
        
        # Check if data is empty
        if data.empty:
            st.error(f"No data found for {ticker} in the specified date range.")
            return None
        
        # Verify that the data contains the necessary columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                st.error(f"Missing required column: {col}")
                return None
        
        # Ensure no NaN values
        if data.isnull().values.any():
            # Fill NaN values with forward fill method
            data = data.fillna(method='ffill')
            # If there are still NaN values at the beginning, use backward fill
            data = data.fillna(method='bfill')
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def get_additional_data(ticker, data):
    """
    Fetch additional market data like company info, sector data, etc.
    This function can be expanded based on requirements.
    
    Parameters:
    ticker (str): Symbol of the stock/cryptocurrency
    data (pandas.DataFrame): Existing price data
    
    Returns:
    dict: Dictionary containing additional data
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        additional_data = {}
        
        # For stocks, get sector and industry info
        if 'sector' in info:
            additional_data['sector'] = info['sector']
        if 'industry' in info:
            additional_data['industry'] = info['industry']
        
        # For cryptocurrencies, could add additional context
        if ticker.endswith('-USD'):
            additional_data['asset_type'] = 'Cryptocurrency'
        else:
            additional_data['asset_type'] = 'Stock'
        
        return additional_data
    
    except Exception as e:
        st.warning(f"Could not fetch additional data: {str(e)}")
        return {}
