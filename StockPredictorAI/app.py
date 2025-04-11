import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from data_collector import fetch_data
from data_preprocessor import preprocess_data, create_sequences, inverse_transform
from model_trainer import train_models
from model_evaluator import evaluate_model
from visualizer import plot_predictions, plot_model_comparison, plot_loss_history
from utils import calculate_metrics

# Set page config
st.set_page_config(
    page_title="Stock & Crypto Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'actual_values' not in st.session_state:
    st.session_state.actual_values = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = {}
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}
if 'prediction_days' not in st.session_state:
    st.session_state.prediction_days = 30

# Title and description
st.title("Stock & Cryptocurrency Price Prediction System")
st.markdown("""
This application uses CNN and LSTM neural networks to predict future prices of stocks and cryptocurrencies.
Select an asset, timeframe, and model parameters to begin.
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
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=1095))  # 3 years by default
end_date = st.sidebar.date_input("End Date", end_date)

# Model parameters
st.sidebar.subheader("Model Parameters")
sequence_length = st.sidebar.slider("Sequence Length", 10, 100, 60)
prediction_days = st.sidebar.slider("Prediction Days", 1, 60, 30)
st.session_state.prediction_days = prediction_days
epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)

# Feature selection
st.sidebar.subheader("Technical Indicators")
use_rsi = st.sidebar.checkbox("RSI", value=True)
use_macd = st.sidebar.checkbox("MACD", value=True)
use_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
use_volume = st.sidebar.checkbox("Volume", value=True)

# Actions
fetch_button = st.sidebar.button("Fetch Data")
train_button = st.sidebar.button("Train Models")

# Main content
tab1, tab2, tab3 = st.tabs(["Data & Predictions", "Model Performance", "Model Comparison"])

with tab1:
    if fetch_button or (selected_asset != st.session_state.selected_asset):
        st.session_state.selected_asset = selected_asset
        
        with st.spinner(f"Fetching data for {selected_asset}..."):
            # Fetch data
            data = fetch_data(selected_asset, start_date, end_date)
            
            if data is not None and not data.empty:
                st.success(f"Data fetched successfully for {selected_asset}")
                
                # Show raw data
                st.subheader("Raw Price Data")
                st.dataframe(data.head())
                
                # Preprocess data
                features = {
                    'rsi': use_rsi,
                    'macd': use_macd,
                    'bollinger': use_bollinger,
                    'volume': use_volume
                }
                
                preprocessed_data, scaler = preprocess_data(data, features)
                
                # Store preprocessed data and scaler in session state
                st.session_state.preprocessed_data[selected_asset] = preprocessed_data
                st.session_state.scalers[selected_asset] = scaler
                
                # Show preprocessed data
                st.subheader("Preprocessed Data with Technical Indicators")
                st.dataframe(preprocessed_data.head())
                
                # Plot historical data
                st.subheader("Historical Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Failed to fetch data for {selected_asset}. Please try again.")
    
    # Display predictions if models are trained
    if selected_asset in st.session_state.predictions and selected_asset in st.session_state.actual_values:
        st.subheader("Price Predictions")
        
        # Plot predictions vs actual values
        plot_predictions(
            st.session_state.actual_values[selected_asset], 
            st.session_state.predictions[selected_asset], 
            selected_asset
        )
        
        # Display metrics if available
        if selected_asset in st.session_state.metrics:
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame(st.session_state.metrics[selected_asset])
            st.dataframe(metrics_df)

with tab2:
    if train_button and selected_asset in st.session_state.preprocessed_data:
        with st.spinner("Training models... This may take a few minutes."):
            # Prepare the data
            preprocessed_data = st.session_state.preprocessed_data[selected_asset]
            scaler = st.session_state.scalers[selected_asset]
            
            # Train the models
            models, history, X_test, y_test = train_models(
                preprocessed_data, 
                sequence_length=sequence_length,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Store models and history in session state
            st.session_state.trained_models[selected_asset] = models
            st.session_state.training_history[selected_asset] = history
            
            # Make predictions
            cnn_pred = models['cnn'].predict(X_test)
            lstm_pred = models['lstm'].predict(X_test)
            
            # Inverse transform predictions and actual values
            cnn_pred_inv = inverse_transform(cnn_pred, scaler)
            lstm_pred_inv = inverse_transform(lstm_pred, scaler)
            y_test_inv = inverse_transform(y_test, scaler)
            
            # Store predictions and actual values
            st.session_state.predictions[selected_asset] = {
                'cnn': cnn_pred_inv,
                'lstm': lstm_pred_inv
            }
            st.session_state.actual_values[selected_asset] = y_test_inv
            
            # Calculate performance metrics
            metrics = {}
            metrics['cnn'] = calculate_metrics(y_test_inv, cnn_pred_inv)
            metrics['lstm'] = calculate_metrics(y_test_inv, lstm_pred_inv)
            st.session_state.metrics[selected_asset] = metrics
            
            st.success("Models trained successfully!")
    
    # Display training history if available
    if selected_asset in st.session_state.training_history:
        st.subheader("Training History")
        plot_loss_history(st.session_state.training_history[selected_asset])
        
        # Show model architecture
        if selected_asset in st.session_state.trained_models:
            st.subheader("CNN Model Architecture")
            st.text(str(st.session_state.trained_models[selected_asset]['cnn'].summary()))
            
            st.subheader("LSTM Model Architecture")
            st.text(str(st.session_state.trained_models[selected_asset]['lstm'].summary()))

with tab3:
    if selected_asset in st.session_state.metrics:
        st.subheader("Model Comparison")
        
        # Compare models
        plot_model_comparison(
            st.session_state.actual_values[selected_asset],
            st.session_state.predictions[selected_asset],
            st.session_state.metrics[selected_asset],
            selected_asset
        )
        
        # Future price prediction
        st.subheader(f"Future Price Prediction (Next {prediction_days} days)")
        
        if st.button("Generate Future Predictions"):
            with st.spinner("Generating future predictions..."):
                # Here we would implement future prediction logic
                # For demonstration, we'll just show a mock future prediction chart
                
                last_date = end_date
                future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
                
                # Plot future predictions
                fig = go.Figure()
                
                # Add historical data
                if selected_asset in st.session_state.preprocessed_data:
                    data = fetch_data(selected_asset, start_date, end_date)
                    fig.add_trace(go.Scatter(
                        x=data.index, 
                        y=data['Close'], 
                        mode='lines',
                        name='Historical Price'
                    ))
                
                # In a real implementation, this would use the trained models to predict
                # future values based on the latest data
                st.info("Note: This is a conceptual representation. In a production environment, the models would generate actual predictions based on the latest data.")
                
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Note:** This application uses historical data for training models. Financial markets are inherently unpredictable, and past performance is not indicative of future results.")
