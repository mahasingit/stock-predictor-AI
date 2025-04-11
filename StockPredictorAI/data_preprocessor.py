import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

def add_technical_indicators(data, features):
    """
    Add technical indicators to the dataset.
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing price data
    features (dict): Dictionary specifying which features to include
    
    Returns:
    pandas.DataFrame: DataFrame with added technical indicators
    """
    df = data.copy()
    
    # Add RSI (Relative Strength Index)
    if features.get('rsi', False):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Add MACD (Moving Average Convergence Divergence)
    if features.get('macd', False):
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Add Bollinger Bands
    if features.get('bollinger', False):
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['SD20'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['SD20'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['SD20'] * 2)
        df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
    
    # Add Volume features
    if features.get('volume', False):
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA10']
    
    # Add basic price-based features
    df['Price_Change'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Drop NaN values that result from the rolling calculations
    df = df.dropna()
    
    return df

def preprocess_data(data, features):
    """
    Preprocess the data by adding technical indicators and normalizing.
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing price data
    features (dict): Dictionary specifying which features to include
    
    Returns:
    tuple: (preprocessed_data, scaler)
    """
    # Add technical indicators
    df = add_technical_indicators(data, features)
    
    # Select relevant columns for modeling
    selected_columns = ['Close', 'Volume', 'Price_Change', 'MA5', 'MA10', 'MA50']
    
    # Add RSI if selected
    if features.get('rsi', False):
        selected_columns.append('RSI')
    
    # Add MACD if selected
    if features.get('macd', False):
        selected_columns.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
    
    # Add Bollinger Bands if selected
    if features.get('bollinger', False):
        selected_columns.extend(['BB_Width'])
    
    # Add Volume features if selected
    if features.get('volume', False):
        selected_columns.extend(['Volume_Change', 'Volume_Ratio'])
    
    # Select only available columns
    selected_columns = [col for col in selected_columns if col in df.columns]
    
    # Create a dataframe with selected features
    preprocessed_data = df[selected_columns].copy()
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    preprocessed_data_scaled = pd.DataFrame(
        scaler.fit_transform(preprocessed_data),
        columns=preprocessed_data.columns,
        index=preprocessed_data.index
    )
    
    return preprocessed_data_scaled, scaler

def create_sequences(data, sequence_length):
    """
    Create sequences for time series prediction.
    
    Parameters:
    data (pandas.DataFrame): Preprocessed DataFrame
    sequence_length (int): Number of time steps in each sequence
    
    Returns:
    tuple: (X, y) where X is input sequences and y is target values
    """
    X, y = [], []
    
    # Convert DataFrame to numpy array
    data_array = data.values
    
    # Create sequences
    for i in range(len(data_array) - sequence_length):
        X.append(data_array[i:i+sequence_length])
        y.append(data_array[i+sequence_length, 0])  # 0 index corresponds to 'Close' price
    
    return np.array(X), np.array(y).reshape(-1, 1)

def train_test_split(X, y, test_size=0.2):
    """
    Split the data into training and testing sets.
    
    Parameters:
    X (numpy.ndarray): Input sequences
    y (numpy.ndarray): Target values
    test_size (float): Proportion of data to use for testing
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def inverse_transform(data, scaler):
    """
    Inverse transform scaled data back to original scale.
    
    Parameters:
    data (numpy.ndarray): Scaled data
    scaler (MinMaxScaler): Scaler used for normalization
    
    Returns:
    numpy.ndarray: Data in original scale
    """
    # Create a dummy array with zeros
    dummy = np.zeros((len(data), scaler.scale_.shape[0]))
    
    # Put the predicted values in the first column (Close price column)
    dummy[:, 0] = data.flatten()
    
    # Inverse transform
    return scaler.inverse_transform(dummy)[:, 0]
