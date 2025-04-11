import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Set page config
st.set_page_config(
    page_title="Stock & Crypto Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'actual_values' not in st.session_state:
    st.session_state.actual_values = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'future_predictions' not in st.session_state:
    st.session_state.future_predictions = {}
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = {}

# Helper functions for predictions
def create_features(data, n_lags=10):
    """Create time series features based on time series lags."""
    df = data.copy()
    
    # Add lag features
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    
    # Add technical indicators
    # SMA
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # EMA
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # RSI (simplified)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    
    # Volume features
    if 'Volume' in df.columns:
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    # Return/price changes
    df['Return'] = df['Close'].pct_change()
    
    # Drop NaN values created by lagging and rolling calculations
    df = df.dropna()
    
    return df

def normalize_data(data):
    """Normalize the data using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Select numerical columns only
    num_cols = data.select_dtypes(include=[np.number]).columns
    normalized_data = data.copy()
    
    # Scale numerical columns
    if len(num_cols) > 0:
        normalized_data[num_cols] = scaler.fit_transform(data[num_cols])
    
    return normalized_data, scaler

def prepare_data_for_models(data, target_col='Close', test_size=0.2):
    """Prepare data for training models."""
    # Get features and target
    X = data.drop(['Close', 'Open', 'High', 'Low', 'Volume', 'Return'], axis=1, errors='ignore')
    y = data[target_col]
    
    # Check for NaN values and remove them
    valid_indices = ~X.isnull().any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Split into train and test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Ensure y is 1D (addresses the warning)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_prediction_models(X_train, y_train):
    """Train multiple ML models for prediction."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            st.warning(f"Could not train {name}: {str(e)}")
    
    return trained_models

def evaluate_prediction_models(models, X_test, y_test, scaler=None, close_scaler=None):
    """Evaluate trained models and return predictions and metrics."""
    predictions = {}
    metrics = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store predictions
        predictions[name] = y_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-10))) * 100
        
        # Calculate directional accuracy
        # Convert to numpy arrays if they're not already
        if hasattr(y_test, 'values'):
            y_test_array = y_test
        else:
            y_test_array = np.array(y_test)
            
        direction_actual = np.sign(y_test_array[1:] - y_test_array[:-1])
        direction_pred = np.sign(y_pred[1:] - y_pred[:-1])
        directional_accuracy = np.mean(direction_actual == direction_pred) * 100
        
        # Store metrics
        metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional Accuracy (%)': directional_accuracy
        }
    
    return predictions, metrics

def make_future_predictions(models, last_data, n_days=30, scaler=None):
    """Generate future predictions using trained models."""
    future_predictions = {}
    
    # Get the last date from the data
    last_date = last_data.index[-1]
    
    # Generate future dates
    future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
    
    # Copy the last row of data to use as a base for predictions
    current_data = last_data.iloc[-1:].copy()
    
    for name, model in models.items():
        # List to store predictions
        model_predictions = []
        
        # Make predictions day by day
        for _ in range(n_days):
            # Get features for prediction (excludes the target variable)
            X_pred = current_data.drop(['Close', 'Open', 'High', 'Low', 'Volume', 'Return'], axis=1, errors='ignore')
            
            # Make prediction
            next_day_pred = model.predict(X_pred)[0]
            model_predictions.append(next_day_pred)
            
            # Update the current data with the new prediction for next step
            # This is a simplified approach - in a real scenario, you would update all features
            current_data['Close'] = next_day_pred
            
            # Update lag features for next prediction
            for lag in range(10, 0, -1):
                if f'lag_{lag}' in current_data.columns:
                    if lag > 1:
                        current_data[f'lag_{lag}'] = current_data[f'lag_{lag-1}']
                    else:
                        current_data[f'lag_1'] = next_day_pred
        
        future_predictions[name] = model_predictions
    
    return future_dates, future_predictions

def plot_prediction_results(actual, predictions, asset_name):
    """Plot actual vs. predicted prices."""
    fig = go.Figure()
    
    # Check if actual is a Series with an index or a numpy array
    if hasattr(actual, 'index'):
        # Use the index if it's a pandas Series
        x_values = actual.index
    else:
        # Create a simple range if it's a numpy array
        x_values = list(range(len(actual)))
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=actual,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add predictions
    colors = ['red', 'green', 'orange', 'purple']
    for i, (model_name, pred) in enumerate(predictions.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=pred,
            mode='lines',
            name=f'{model_name} Prediction',
            line=dict(color=color, width=2, dash='dot')
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{asset_name} - Actual vs Predicted Prices",
        xaxis_title="Data Points" if not hasattr(actual, 'index') else "Date",
        yaxis_title="Price",
        legend_title="Legend",
        height=500
    )
    
    return fig

def plot_future_predictions(last_date, last_price, future_dates, predictions, asset_name):
    """Plot future price predictions."""
    fig = go.Figure()
    
    # Add marker for last known price
    fig.add_trace(go.Scatter(
        x=[last_date],
        y=[last_price],
        mode='markers',
        name='Last Known Price',
        marker=dict(color='blue', size=10)
    ))
    
    # Add predictions
    colors = ['red', 'green', 'orange', 'purple']
    for i, (model_name, pred) in enumerate(predictions.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pred,
            mode='lines+markers',
            name=f'{model_name} Prediction',
            line=dict(color=color, width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{asset_name} - Future Price Predictions",
        xaxis_title="Date",
        yaxis_title="Predicted Price",
        legend_title="Models",
        height=500
    )
    
    return fig

def plot_model_comparison(metrics):
    """Create visualizations comparing model performance."""
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame({
        model: {metric: value for metric, value in model_metrics.items()} 
        for model, model_metrics in metrics.items()
    }).T
    
    # Select important metrics for visualization
    key_metrics = ['RMSE', 'MAE', 'MAPE', 'Directional Accuracy (%)']
    selected_metrics = metrics_df[key_metrics]
    
    # Create a figure with subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=key_metrics,
                        specs=[[{'type': 'bar'}, {'type': 'bar'}],
                              [{'type': 'bar'}, {'type': 'bar'}]])
    
    # Colors for the models
    colors = ['red', 'green', 'orange', 'purple']
    
    # Add traces for each metric
    for i, metric in enumerate(key_metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(
            go.Bar(
                x=selected_metrics.index,
                y=selected_metrics[metric],
                marker_color=colors[:len(selected_metrics)],
                name=metric
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title="Model Performance Comparison",
        height=700,
        showlegend=False
    )
    
    return fig

# Title and description
st.title("Stock & Cryptocurrency Price Prediction System")
st.markdown("""
This application uses machine learning to predict future prices of stocks and cryptocurrencies.
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
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365*2))  # 2 years by default
end_date = st.sidebar.date_input("End Date", end_date)

# Model parameters
st.sidebar.subheader("Model Parameters")
n_lags = st.sidebar.slider("Number of Lag Features", 5, 30, 10)
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
prediction_days = st.sidebar.slider("Future Prediction Days", 7, 60, 30)

# Actions
fetch_button = st.sidebar.button("Fetch Data")
train_button = st.sidebar.button("Train Models")
predict_button = st.sidebar.button("Generate Future Predictions")

# Import sentiment analyzer
from sentiment_analyzer import perform_combined_sentiment_analysis, plot_sentiment_over_time, plot_sentiment_distribution, SentimentAnalyzer

# Import stock mood module
from stock_mood import get_prediction_mood_emoji, get_sentiment_mood_emoji, display_stock_mood, get_overall_stock_mood, get_combined_mood_emoji

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Data & Analysis", "Model Performance", "Future Predictions", "Sentiment Analysis"])

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
                    
                    # Add technical indicators
                    with st.spinner("Preparing data and adding technical indicators..."):
                        # Create features for analysis
                        feature_data = create_features(data, n_lags=n_lags)
                        
                        # Store in session state for later use
                        st.session_state.feature_data = feature_data
                    
                    # Plot historical data
                    st.subheader("Historical Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                    
                    # Add SMAs
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), 
                                         mode='lines', name='SMA 20', line=dict(dash='dash')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=50).mean(), 
                                         mode='lines', name='SMA 50', line=dict(dash='dash')))
                    
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
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                        daily_change = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100
                        st.metric("Daily Change", f"{daily_change:.2f}%")
                    
                    with col2:
                        st.metric("Highest Price", f"${data['High'].max():.2f}")
                        st.metric("Lowest Price", f"${data['Low'].min():.2f}")
                    
                    with col3:
                        returns = data['Close'].pct_change().dropna()
                        st.metric("Volatility (Std Dev)", f"{returns.std() * 100:.2f}%")
                        st.metric("Avg Daily Return", f"{returns.mean() * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

with tab2:
    if train_button and 'feature_data' in st.session_state:
        with st.spinner("Training prediction models... This may take a moment."):
            try:
                # Get data
                feature_data = st.session_state.feature_data
                
                # Prepare data for models
                X_train, X_test, y_train, y_test = prepare_data_for_models(
                    feature_data, 
                    target_col='Close', 
                    test_size=test_size
                )
                
                # Train models
                trained_models = train_prediction_models(X_train, y_train)
                
                # Store models in session state
                st.session_state.trained_models[selected_asset] = trained_models
                
                # Evaluate models
                predictions, metrics = evaluate_prediction_models(
                    trained_models, 
                    X_test, 
                    y_test
                )
                
                # Store results in session state
                st.session_state.predictions[selected_asset] = predictions
                st.session_state.actual_values[selected_asset] = y_test
                st.session_state.metrics[selected_asset] = metrics
                
                st.success("Models trained successfully!")
                
                # Display prediction results
                st.subheader("Price Prediction Results")
                fig = plot_prediction_results(y_test, predictions, selected_asset)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display performance metrics
                st.subheader("Model Performance Metrics")
                metrics_df = pd.DataFrame({model: metrics for model, metrics in metrics.items()}).T
                st.dataframe(metrics_df)
                
                # Display model comparison
                st.subheader("Model Comparison")
                fig_comp = plot_model_comparison(metrics)
                st.plotly_chart(fig_comp, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
    
    elif selected_asset in st.session_state.predictions:
        # Display previously generated predictions
        st.subheader("Price Prediction Results")
        fig = plot_prediction_results(
            st.session_state.actual_values[selected_asset], 
            st.session_state.predictions[selected_asset], 
            selected_asset
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display performance metrics
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame({model: metrics for model, metrics in st.session_state.metrics[selected_asset].items()}).T
        st.dataframe(metrics_df)
        
        # Display model comparison
        st.subheader("Model Comparison")
        fig_comp = plot_model_comparison(st.session_state.metrics[selected_asset])
        st.plotly_chart(fig_comp, use_container_width=True)
    
    else:
        st.info("Please fetch data and train models first.")

with tab3:
    if predict_button and selected_asset in st.session_state.trained_models:
        with st.spinner(f"Generating future price predictions for {selected_asset}..."):
            try:
                # Get data and models
                feature_data = st.session_state.feature_data
                trained_models = st.session_state.trained_models[selected_asset]
                
                # Generate future predictions
                future_dates, future_predictions = make_future_predictions(
                    trained_models, 
                    feature_data, 
                    n_days=prediction_days
                )
                
                # Store in session state
                st.session_state.future_predictions[selected_asset] = {
                    'dates': future_dates,
                    'predictions': future_predictions
                }
                
                st.success(f"Generated future predictions for the next {prediction_days} days!")
                
                # Plot future predictions
                st.subheader(f"Future Price Predictions for {selected_asset}")
                last_date = feature_data.index[-1]
                last_price = feature_data['Close'].iloc[-1]
                
                # Add stock mood emoji indicator
                st.subheader("Stock Mood")
                
                # Get prediction mood based on average prediction for first day
                try:
                    avg_first_day_pred = sum([preds[0] for model, preds in future_predictions.items() if len(preds) > 0]) / len(future_predictions)
                    prediction_mood = get_prediction_mood_emoji(avg_first_day_pred, last_price)
                except (IndexError, ZeroDivisionError):
                    # Fallback if there's an issue with the predictions
                    prediction_mood = ("❓", "Unknown", "#CCCCCC")
                
                # Get sentiment mood if available
                sentiment_mood = None
                if selected_asset in st.session_state.sentiment_data:
                    sentiment_score = st.session_state.sentiment_data[selected_asset]['overall_sentiment']
                    sentiment_mood = get_sentiment_mood_emoji(sentiment_score)
                
                # Display stock mood
                display_stock_mood(prediction_mood, sentiment_mood)
                
                # Get overall mood (combining prediction and sentiment)
                if sentiment_mood:
                    overall_mood = get_overall_stock_mood(prediction_mood, sentiment_mood)
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 10px;">
                        <p>Overall recommendation: <span style="font-size: 1.2em; font-weight: bold; color: {overall_mood[2]};">{overall_mood[1]} {overall_mood[0]}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                fig = plot_future_predictions(
                    last_date, 
                    last_price, 
                    future_dates, 
                    future_predictions, 
                    selected_asset
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction in table format
                st.subheader("Predicted Prices by Model")
                
                # Create a DataFrame with predictions from each model
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    **{model: preds for model, preds in future_predictions.items()}
                }).set_index('Date')
                
                # Calculate average prediction
                pred_df['Average Prediction'] = pred_df.mean(axis=1)
                
                # Display table
                st.dataframe(pred_df)
                
                # Show prediction accuracy metrics
                st.subheader("Model Accuracy for Price Prediction")
                if selected_asset in st.session_state.metrics:
                    acc_df = pd.DataFrame()
                    for model, metrics in st.session_state.metrics[selected_asset].items():
                        acc_df[model] = [
                            f"{metrics['MAPE']:.2f}%",
                            f"{metrics['Directional Accuracy (%)']:.2f}%"
                        ]
                    
                    acc_df.index = ['Price Accuracy (MAPE)', 'Direction Accuracy']
                    st.table(acc_df)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
    
    elif selected_asset in st.session_state.future_predictions:
        # Display previously generated future predictions
        future_data = st.session_state.future_predictions[selected_asset]
        
        st.subheader(f"Future Price Predictions for {selected_asset}")
        
        # Get last known price (if possible)
        if 'feature_data' in st.session_state:
            last_date = st.session_state.feature_data.index[-1]
            last_price = st.session_state.feature_data['Close'].iloc[-1]
        else:
            # Fallback if feature data is not available
            data = yf.download(selected_asset, start=(datetime.now()-timedelta(days=5)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
            last_date = data.index[-1]
            last_price = data['Close'].iloc[-1]
        
        # Add stock mood emoji indicator
        st.subheader("Stock Mood")
        
        # Get prediction mood based on average prediction for first day
        predictions_dict = future_data['predictions']
        try:
            avg_first_day_pred = sum([preds[0] for model, preds in predictions_dict.items() if len(preds) > 0]) / len(predictions_dict)
            prediction_mood = get_prediction_mood_emoji(avg_first_day_pred, last_price)
        except (IndexError, ZeroDivisionError):
            # Fallback if there's an issue with the predictions
            prediction_mood = ("❓", "Unknown", "#CCCCCC")
        
        # Get sentiment mood if available
        sentiment_mood = None
        if selected_asset in st.session_state.sentiment_data:
            sentiment_score = st.session_state.sentiment_data[selected_asset]['overall_sentiment']
            sentiment_mood = get_sentiment_mood_emoji(sentiment_score)
        
        # Display stock mood
        display_stock_mood(prediction_mood, sentiment_mood)
        
        # Get overall mood (combining prediction and sentiment)
        if sentiment_mood:
            overall_mood = get_overall_stock_mood(prediction_mood, sentiment_mood)
            st.markdown(f"""
            <div style="text-align: center; margin-top: 10px;">
                <p>Overall recommendation: <span style="font-size: 1.2em; font-weight: bold; color: {overall_mood[2]};">{overall_mood[1]} {overall_mood[0]}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        fig = plot_future_predictions(
            last_date, 
            last_price, 
            future_data['dates'], 
            future_data['predictions'], 
            selected_asset
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction in table format
        st.subheader("Predicted Prices by Model")
        
        # Create a DataFrame with predictions from each model
        pred_df = pd.DataFrame({
            'Date': future_data['dates'],
            **{model: preds for model, preds in future_data['predictions'].items()}
        }).set_index('Date')
        
        # Calculate average prediction
        pred_df['Average Prediction'] = pred_df.mean(axis=1)
        
        # Display table
        st.dataframe(pred_df)
        
        # Show prediction accuracy metrics
        st.subheader("Model Accuracy for Price Prediction")
        if selected_asset in st.session_state.metrics:
            acc_df = pd.DataFrame()
            for model, metrics in st.session_state.metrics[selected_asset].items():
                acc_df[model] = [
                    f"{metrics['MAPE']:.2f}%",
                    f"{metrics['Directional Accuracy (%)']:.2f}%"
                ]
            
            acc_df.index = ['Price Accuracy (MAPE)', 'Direction Accuracy']
            st.table(acc_df)
    
    else:
        st.info("Please fetch data, train models, and then generate future predictions.")

# Add sentiment analysis tab content
with tab4:
    st.subheader(f"Market Sentiment Analysis for {selected_asset}")
    
    # Initialize session state for sentiment if it doesn't exist
    if 'sentiment_data' not in st.session_state:
        st.session_state.sentiment_data = {}
    
    # Add time period selection
    sentiment_time_days = st.slider("Analysis Period (Days)", 1, 30, 7)
    
    # Add refresh button
    sentiment_refresh = st.button("Analyze Market Sentiment")
    
    if sentiment_refresh or (selected_asset not in st.session_state.sentiment_data):
        with st.spinner(f"Analyzing market sentiment for {selected_asset}..."):
            try:
                # Perform sentiment analysis
                sentiment_results = perform_combined_sentiment_analysis(selected_asset, sentiment_time_days)
                
                # Store in session state
                st.session_state.sentiment_data[selected_asset] = sentiment_results
                
                st.success(f"Sentiment analysis completed for {selected_asset}!")
            except Exception as e:
                st.error(f"Error analyzing sentiment: {str(e)}")
    
    # Display sentiment analysis results if available
    if selected_asset in st.session_state.sentiment_data:
        sentiment_data = st.session_state.sentiment_data[selected_asset]
        
        # Overall sentiment
        sentiment_analyzer = SentimentAnalyzer()
        overall_score = sentiment_data['overall_sentiment']
        overall_label = sentiment_data['overall_label']
        sentiment_color = sentiment_analyzer.get_sentiment_color(overall_score)
        
        # Display overall sentiment
        st.subheader("Overall Market Sentiment")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background-color: {sentiment_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h1 style="margin: 0; color: white;">{overall_label}</h1>
                <p style="margin: 5px 0 0 0; color: white;">Score: {overall_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("News Sentiment", f"{sentiment_data['news_sentiment']:.2f}", 
                     delta=f"{sentiment_data['news_sentiment'] - sentiment_data['social_sentiment']:.2f}")
        
        with col3:
            st.metric("Social Media Sentiment", f"{sentiment_data['social_sentiment']:.2f}")
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        dist_fig = plot_sentiment_distribution(sentiment_data['combined_distribution'], 
                                              title="Overall Sentiment Distribution")
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # News sentiment over time
        if sentiment_data['news_sentiment_time']:
            st.subheader("News Sentiment Over Time")
            time_fig = plot_sentiment_over_time(sentiment_data['news_sentiment_time'], 
                                               title=f"{selected_asset} News Sentiment Trend")
            st.plotly_chart(time_fig, use_container_width=True)
        
        # News articles
        st.subheader("Recent News Articles")
        for i, article in enumerate(sentiment_data['news_articles'][:5]):  # Show top 5 articles
            sentiment_color = sentiment_analyzer.get_sentiment_color(article['sentiment_score'])
            
            st.markdown(f"""
            <div style="border-left: 5px solid {sentiment_color}; padding-left: 10px; margin-bottom: 15px;">
                <h4 style="margin: 0;">{article['title']}</h4>
                <p style="margin: 5px 0; font-size: 0.8em; color: gray;">
                    Source: {article['source']} | Date: {article['date'].strftime('%Y-%m-%d')}
                </p>
                <p style="margin: 0;">Sentiment: {article['sentiment_label']} ({article['sentiment_score']:.2f})</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Social media posts
        st.subheader("Social Media Sentiment")
        for i, post in enumerate(sentiment_data['social_posts'][:5]):  # Show top 5 posts
            sentiment_color = sentiment_analyzer.get_sentiment_color(post['sentiment'])
            
            st.markdown(f"""
            <div style="border-left: 5px solid {sentiment_color}; padding-left: 10px; margin-bottom: 15px;">
                <p style="margin: 0;">{post['content']}</p>
                <p style="margin: 5px 0; font-size: 0.8em; color: gray;">
                    Platform: {post['platform']} | {post['timestamp'].strftime('%Y-%m-%d %H:%M')}
                </p>
                <p style="margin: 0;">Sentiment: {post['category'].capitalize()} ({post['sentiment']:.2f})</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sentiment impact on price
        st.subheader("Sentiment Impact on Price Predictions")
        
        # Show disclaimer about sentiment analysis
        st.info("""
        The sentiment analysis shown above uses NLTK and VADER for natural language processing. 
        This analyzes the tone and content of news articles and social media posts to assess market sentiment.
        Positive sentiment often (but not always) correlates with price increases, while negative sentiment may indicate downward pressure.
        
        **Note:** The social media data shown is synthetic for demonstration purposes. In a production environment, 
        this would be replaced with real data from Twitter, Reddit, or StockTwits APIs.
        """)
    else:
        st.info("Please click 'Analyze Market Sentiment' to see sentiment analysis for this asset.")
        
# Footer
st.markdown("---")
st.markdown("""
**Note:** This application uses machine learning models trained on historical data for price prediction. 
Financial markets are inherently unpredictable, and all predictions should be considered as estimates rather than guarantees. 
Past performance is not indicative of future results.
""")