import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, scaler=None):
    """
    Evaluate the performance of a model.
    
    Parameters:
    model (tensorflow.keras.Model): Trained model
    X_test (numpy.ndarray): Test input data
    y_test (numpy.ndarray): Test target values
    scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for normalization
    
    Returns:
    dict: Dictionary containing performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # If a scaler is provided, inverse transform the data
    if scaler is not None:
        # Create a dummy array with zeros
        dummy_y_test = np.zeros((len(y_test), scaler.scale_.shape[0]))
        dummy_y_pred = np.zeros((len(y_pred), scaler.scale_.shape[0]))
        
        # Put the values in the first column (Close price column)
        dummy_y_test[:, 0] = y_test.flatten()
        dummy_y_pred[:, 0] = y_pred.flatten()
        
        # Inverse transform
        y_test_inv = scaler.inverse_transform(dummy_y_test)[:, 0]
        y_pred_inv = scaler.inverse_transform(dummy_y_pred)[:, 0]
    else:
        y_test_inv = y_test.flatten()
        y_pred_inv = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    # Calculate directional accuracy
    direction_correct = np.sum(np.sign(y_test_inv[1:] - y_test_inv[:-1]) == 
                              np.sign(y_pred_inv[1:] - y_pred_inv[:-1]))
    directional_accuracy = direction_correct / (len(y_test_inv) - 1) * 100
    
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
