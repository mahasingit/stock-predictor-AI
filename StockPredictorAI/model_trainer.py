import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

from model_cnn import build_cnn_model
from model_lstm import build_lstm_model
from data_preprocessor import create_sequences, train_test_split

def train_models(data, sequence_length=60, test_size=0.2, epochs=50, batch_size=32):
    """
    Train CNN and LSTM models on the provided data.
    
    Parameters:
    data (pandas.DataFrame): Preprocessed data
    sequence_length (int): Number of time steps in each sequence
    test_size (float): Proportion of data to use for testing
    epochs (int): Number of training epochs
    batch_size (int): Batch size for training
    
    Returns:
    tuple: (models, history, X_test, y_test)
    """
    # Create sequences
    X, y = create_sequences(data, sequence_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Build and train CNN model
    st.text("Training CNN model...")
    cnn_model = build_cnn_model(input_shape)
    cnn_history = cnn_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Build and train LSTM model
    st.text("Training LSTM model...")
    lstm_model = build_lstm_model(input_shape)
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Combine models and history
    models = {
        'cnn': cnn_model,
        'lstm': lstm_model
    }
    
    history = {
        'cnn': cnn_history.history,
        'lstm': lstm_history.history
    }
    
    return models, history, X_test, y_test
