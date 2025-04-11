import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_model(input_shape, dropout_rate=0.2):
    """
    Build a CNN model for time series prediction.
    
    Parameters:
    input_shape (tuple): Shape of the input data (sequence_length, n_features)
    dropout_rate (float): Dropout rate for regularization
    
    Returns:
    tensorflow.keras.models.Sequential: Compiled CNN model
    """
    model = Sequential([
        # First convolutional layer
        Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same',
            input_shape=input_shape
        ),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        # Second convolutional layer
        Conv1D(
            filters=128,
            kernel_size=3,
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        # Third convolutional layer
        Conv1D(
            filters=256,
            kernel_size=3,
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model
