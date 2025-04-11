import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional

def build_lstm_model(input_shape, dropout_rate=0.2):
    """
    Build an LSTM model for time series prediction.
    
    Parameters:
    input_shape (tuple): Shape of the input data (sequence_length, n_features)
    dropout_rate (float): Dropout rate for regularization
    
    Returns:
    tensorflow.keras.models.Sequential: Compiled LSTM model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=input_shape
        ),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        LSTM(
            units=100,
            return_sequences=True
        ),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Third LSTM layer with bidirectional wrapper
        Bidirectional(
            LSTM(
                units=50,
                return_sequences=False
            )
        ),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model
