import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="Stock & Crypto Price Analysis",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = None

# Title and description
st.title("Stock & Cryptocurrency Price Analysis System")
st.markdown("""
This application analyzes price data for stocks and cryptocurrencies using historical data from Yahoo Finance.
Select an asset and timeframe to begin.
""")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Asset selection
asset_type = st.sidebar.radio("Asset Type", ["Stocks", "Cryptocurrencies"])

# Predefined lists of assets
stock_options = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "BAC", "V"]
crypto_options = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LINK-USD"]

# Asset selection based on asset type
if asset_type == "Stocks":
    selected_asset = st.sidebar.selectbox("Select a Stock", stock_options)
else:
    selected_asset = st.sidebar.selectbox("Select a Cryptocurrency", crypto_options)

# Date range selection
st.sidebar.subheader("Date Range")
end_date = datetime.now().date()
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))  # 1 year by default
end_date = st.sidebar.date_input("End Date", end_date)

# Technical indicators
st.sidebar.subheader("Technical Indicators")
use_sma = st.sidebar.checkbox("Simple Moving Averages", value=True)
sma_periods = [20, 50, 200] if use_sma else []

# Actions
fetch_button = st.sidebar.button("Fetch Data")

# Main content
tab1, tab2 = st.tabs(["Price Data", "Technical Analysis"])

with tab1:
    if fetch_button or (selected_asset != st.session_state.selected_asset):
        st.session_state.selected_asset = selected_asset
        
        with st.spinner(f"Fetching data for {selected_asset}..."):
            try:
                # Convert dates to string format for yfinance
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # Fetch data from Yahoo Finance
                data = yf.download(selected_asset, start=start_str, end=end_str)
                
                if data.empty:
                    st.error(f"No data found for {selected_asset} in the specified date range.")
                else:
                    st.success(f"Data fetched successfully for {selected_asset}")
                    
                    # Show raw data
                    st.subheader("Raw Price Data")
                    st.dataframe(data.head())
                    
                    # Calculate returns
                    data['Daily Return'] = data['Close'].pct_change() * 100
                    
                    # Add SMAs if selected
                    if use_sma:
                        for period in sma_periods:
                            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
                    
                    # Plot historical data
                    st.subheader("Historical Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                    
                    # Add SMAs to the plot
                    if use_sma:
                        for period in sma_periods:
                            fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA_{period}'], 
                                                  mode='lines', name=f'SMA {period}',
                                                  line=dict(dash='dash')))
                    
                    fig.update_layout(
                        title=f"{selected_asset} Price History",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    st.subheader("Trading Volume")
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
                    fig_volume.update_layout(
                        title=f"{selected_asset} Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        height=400
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)
                    
                    # Price statistics
                    st.subheader("Price Statistics")
                    stats = {
                        "Current Price": data['Close'].iloc[-1],
                        "Change (1 Day)": f"{data['Close'].iloc[-1] - data['Close'].iloc[-2]:.2f} ({(data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100:.2f}%)",
                        "Highest Price": data['High'].max(),
                        "Lowest Price": data['Low'].min(),
                        "Average Price": data['Close'].mean(),
                        "Price Volatility": data['Daily Return'].std()
                    }
                    
                    # Create 2 columns for stats
                    col1, col2 = st.columns(2)
                    
                    for i, (key, value) in enumerate(stats.items()):
                        if i < 3:
                            col1.metric(key, value)
                        else:
                            col2.metric(key, value)
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

with tab2:
    if fetch_button or (selected_asset != st.session_state.selected_asset):
        try:
            if 'data' in locals() and not data.empty:
                st.subheader("Technical Analysis")
                
                # Daily returns
                st.subheader("Daily Returns Distribution")
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Histogram(x=data['Daily Return'].dropna(), nbinsx=50))
                fig_returns.update_layout(
                    title=f"{selected_asset} Daily Returns Distribution",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig_returns, use_container_width=True)
                
                # Rolling statistics
                st.subheader("Rolling Statistics")
                window_size = st.slider("Rolling Window Size (days)", 7, 100, 20)
                
                # Calculate rolling statistics
                data['Rolling Mean'] = data['Close'].rolling(window=window_size).mean()
                data['Rolling Std'] = data['Close'].rolling(window=window_size).std()
                
                # Plot rolling statistics
                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                fig_rolling.add_trace(go.Scatter(x=data.index, y=data['Rolling Mean'], mode='lines', 
                                              name=f'Rolling Mean ({window_size} days)',
                                              line=dict(color='orange')))
                fig_rolling.add_trace(go.Scatter(x=data.index, y=data['Rolling Mean'] + 2*data['Rolling Std'], 
                                              mode='lines', name='Upper Band', line=dict(dash='dash')))
                fig_rolling.add_trace(go.Scatter(x=data.index, y=data['Rolling Mean'] - 2*data['Rolling Std'], 
                                              mode='lines', name='Lower Band', line=dict(dash='dash')))
                fig_rolling.update_layout(
                    title=f"{selected_asset} Price with Rolling Statistics",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=500
                )
                st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Correlation with market (if stock)
                if asset_type == "Stocks":
                    try:
                        st.subheader("Correlation with Market")
                        market_data = yf.download("SPY", start=start_str, end=end_str)
                        if not market_data.empty:
                            # Calculate returns
                            market_data['Market Return'] = market_data['Close'].pct_change() * 100
                            
                            # Merge data
                            merged_data = pd.DataFrame()
                            merged_data[f'{selected_asset} Return'] = data['Daily Return']
                            merged_data['Market Return'] = market_data['Market Return']
                            merged_data = merged_data.dropna()
                            
                            # Calculate correlation
                            correlation = merged_data[f'{selected_asset} Return'].corr(merged_data['Market Return'])
                            
                            # Display correlation
                            st.metric("Correlation with S&P 500", f"{correlation:.4f}")
                            
                            # Scatter plot
                            fig_corr = go.Figure()
                            fig_corr.add_trace(go.Scatter(x=merged_data['Market Return'], y=merged_data[f'{selected_asset} Return'], 
                                                       mode='markers'))
                            fig_corr.update_layout(
                                title=f"Correlation between {selected_asset} and S&P 500",
                                xaxis_title="S&P 500 Daily Return (%)",
                                yaxis_title=f"{selected_asset} Daily Return (%)",
                                height=400
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in correlation analysis: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error in technical analysis: {str(e)}")
            if 'data' not in locals() or data.empty:
                st.info("Please fetch data first to perform technical analysis.")

# Footer
st.markdown("---")
st.markdown("**Note:** This application uses historical data for analysis. Financial markets are inherently unpredictable, and past performance is not indicative of future results.")