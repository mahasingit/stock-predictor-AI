import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics for model evaluation.
    
    Parameters:
    y_true (numpy.ndarray): True values
    y_pred (numpy.ndarray): Predicted values
    
    Returns:
    dict: Dictionary containing performance metrics
    """
    # Ensure arrays are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100
    
    # Calculate directional accuracy
    # Checks if the direction of change is the same for actual and predicted values
    direction_correct = np.sum(np.sign(y_true[1:] - y_true[:-1]) == 
                              np.sign(y_pred[1:] - y_pred[:-1]))
    directional_accuracy = direction_correct / (len(y_true) - 1) * 100
    
    # Return metrics
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Directional Accuracy (%)': directional_accuracy
    }
    
    return metrics

def create_future_sequences(last_sequence, scaler, num_days=30):
    """
    Create input sequences for future predictions.
    
    Parameters:
    last_sequence (numpy.ndarray): Last sequence from the training data
    scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for normalization
    num_days (int): Number of days to predict
    
    Returns:
    numpy.ndarray: Array of future sequences
    """
    future_sequences = []
    current_sequence = last_sequence.copy()
    
    for _ in range(num_days):
        future_sequences.append(current_sequence)
        # Shift the sequence for the next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        # The last value is unknown, set it to the previous value for now
        current_sequence[-1] = current_sequence[-2]
    
    return np.array(future_sequences)

def format_large_number(num):
    """
    Format large numbers for display.
    
    Parameters:
    num (float): Number to format
    
    Returns:
    str: Formatted number
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"
