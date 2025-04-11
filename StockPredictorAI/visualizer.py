import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_predictions(actual, predictions, asset_name):
    """
    Plot actual prices vs model predictions.
    
    Parameters:
    actual (numpy.ndarray): Actual price values
    predictions (dict): Dictionary containing predictions from different models
    asset_name (str): Name of the asset
    """
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        y=actual,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add CNN predictions
    if 'cnn' in predictions:
        fig.add_trace(go.Scatter(
            y=predictions['cnn'],
            mode='lines',
            name='CNN Prediction',
            line=dict(color='red', width=2, dash='dot')
        ))
    
    # Add LSTM predictions
    if 'lstm' in predictions:
        fig.add_trace(go.Scatter(
            y=predictions['lstm'],
            mode='lines',
            name='LSTM Prediction',
            line=dict(color='green', width=2, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{asset_name} - Actual vs Predicted Prices",
        xaxis_title="Time",
        yaxis_title="Price",
        legend_title="Legend",
        height=500
    )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)

def plot_loss_history(history):
    """
    Plot training and validation loss history.
    
    Parameters:
    history (dict): Dictionary containing training history
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("CNN Model Loss", "LSTM Model Loss")
    )
    
    # Add CNN loss traces
    if 'cnn' in history:
        fig.add_trace(
            go.Scatter(
                y=history['cnn']['loss'],
                mode='lines',
                name='CNN Training Loss',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=history['cnn']['val_loss'],
                mode='lines',
                name='CNN Validation Loss',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
    
    # Add LSTM loss traces
    if 'lstm' in history:
        fig.add_trace(
            go.Scatter(
                y=history['lstm']['loss'],
                mode='lines',
                name='LSTM Training Loss',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                y=history['lstm']['val_loss'],
                mode='lines',
                name='LSTM Validation Loss',
                line=dict(color='green', dash='dash')
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Model Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400
    )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)

def plot_model_comparison(actual, predictions, metrics, asset_name):
    """
    Create a comparison visualization of model performance.
    
    Parameters:
    actual (numpy.ndarray): Actual price values
    predictions (dict): Dictionary containing predictions from different models
    metrics (dict): Dictionary containing performance metrics
    asset_name (str): Name of the asset
    """
    # Create a metrics comparison chart
    metrics_df = pd.DataFrame({
        'Metric': list(metrics['cnn'].keys()),
        'CNN': list(metrics['cnn'].values()),
        'LSTM': list(metrics['lstm'].values())
    })
    
    # Display metrics in a table
    st.dataframe(metrics_df.set_index('Metric'))
    
    # Create a bar chart for key metrics
    key_metrics = ['RMSE', 'MAE', 'MAPE', 'Directional Accuracy (%)']
    filtered_df = metrics_df[metrics_df['Metric'].isin(key_metrics)]
    
    # Reshape for plotting
    plot_df = pd.melt(
        filtered_df, 
        id_vars=['Metric'], 
        value_vars=['CNN', 'LSTM'],
        var_name='Model', 
        value_name='Value'
    )
    
    # Create subplots
    fig = make_subplots(rows=1, cols=len(key_metrics), subplot_titles=key_metrics)
    
    # Add bars for each metric
    for i, metric in enumerate(key_metrics):
        metric_df = plot_df[plot_df['Metric'] == metric]
        
        fig.add_trace(
            go.Bar(
                x=metric_df['Model'],
                y=metric_df['Value'],
                name=metric,
                marker_color=['red', 'green'],
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{asset_name} - Model Performance Comparison",
        height=400
    )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Create error analysis plot
    st.subheader("Prediction Error Analysis")
    
    # Calculate errors
    cnn_errors = actual - predictions['cnn']
    lstm_errors = actual - predictions['lstm']
    
    # Create error plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=cnn_errors,
        mode='lines',
        name='CNN Error',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        y=lstm_errors,
        mode='lines',
        name='LSTM Error',
        line=dict(color='green')
    ))
    
    # Add a zero line
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=len(cnn_errors),
        y1=0,
        line=dict(
            color="black",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Prediction Error Over Time",
        xaxis_title="Time",
        yaxis_title="Error (Actual - Predicted)",
        height=400
    )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
