import streamlit as st

def get_prediction_mood_emoji(prediction, current_price, threshold=1.0):
    """
    Return an appropriate emoji based on the prediction compared to the current price.
    
    Parameters:
    prediction (float or pandas.Series): Predicted price
    current_price (float or pandas.Series): Current price
    threshold (float): Threshold percentage for significant change
    
    Returns:
    tuple: (emoji, description, color)
    """
    # Convert to float if pandas Series or DataFrame
    try:
        if hasattr(prediction, 'iloc'):
            prediction = float(prediction.iloc[0])
        else:
            prediction = float(prediction)
            
        if hasattr(current_price, 'iloc'):
            current_price = float(current_price.iloc[0])
        else:
            current_price = float(current_price)
    except (ValueError, IndexError, TypeError) as e:
        # If conversion fails, return unknown mood
        return "❓", "Unknown", "#CCCCCC"
    
    # Calculate percent change
    try:
        percent_change = ((prediction / current_price) - 1) * 100
    except (ZeroDivisionError, TypeError):
        return "❓", "Unknown", "#CCCCCC"
    
    # Very bullish (>5%)
    if percent_change > 5:
        return "🚀", "Very bullish", "#00CC00"
    # Bullish (2-5%)
    elif percent_change > 2:
        return "😁", "Bullish", "#66CC66" 
    # Slightly bullish (0.5-2%)
    elif percent_change > 0.5:
        return "🙂", "Slightly bullish", "#99CC99"
    # Neutral (-0.5 to 0.5%)
    elif percent_change >= -0.5:
        return "😐", "Neutral", "#CCCCCC"
    # Slightly bearish (-2 to -0.5%)
    elif percent_change >= -2:
        return "😕", "Slightly bearish", "#CC9999"
    # Bearish (-5 to -2%)
    elif percent_change >= -5:
        return "😟", "Bearish", "#CC6666"
    # Very bearish (<-5%)
    else:
        return "🧸", "Very bearish", "#CC0000"

def get_combined_mood_emoji(predictions, current_price):
    """
    Return a combined mood emoji based on the average of all predictions.
    
    Parameters:
    predictions (dict): Dictionary of model predictions
    current_price (float or pandas.Series): Current price
    
    Returns:
    tuple: (emoji, description, color)
    """
    if not predictions:
        return "❓", "Unknown", "#CCCCCC"
    
    # Convert current_price to float if pandas Series or DataFrame
    try:
        if hasattr(current_price, 'iloc'):
            current_price = float(current_price.iloc[0])
        else:
            current_price = float(current_price)
    except (ValueError, IndexError, TypeError):
        # If conversion fails, return unknown mood
        return "❓", "Unknown", "#CCCCCC"
    
    # Calculate the average prediction
    all_predictions = []
    for model, preds in predictions.items():
        if len(preds) > 0:
            try:
                all_predictions.append(float(preds[0]))  # Use the first prediction and convert to float
            except (ValueError, TypeError):
                # Skip if conversion fails
                continue
    
    if not all_predictions:
        return "❓", "Unknown", "#CCCCCC"
    
    avg_prediction = sum(all_predictions) / len(all_predictions)
    return get_prediction_mood_emoji(avg_prediction, current_price)

def get_sentiment_mood_emoji(sentiment_score):
    """
    Return an appropriate emoji based on the sentiment score.
    
    Parameters:
    sentiment_score (float or pandas.Series): Sentiment score from -1 to 1
    
    Returns:
    tuple: (emoji, description, color)
    """
    # Convert to float if pandas Series or DataFrame
    try:
        if hasattr(sentiment_score, 'iloc'):
            sentiment_score = float(sentiment_score.iloc[0])
        else:
            sentiment_score = float(sentiment_score)
    except (ValueError, IndexError, TypeError):
        # If conversion fails, return neutral mood
        return "😐", "Neutral", "#CCCCCC"
    
    if sentiment_score > 0.5:
        return "😍", "Very positive", "#00CC00"
    elif sentiment_score > 0.2:
        return "😊", "Positive", "#66CC66"
    elif sentiment_score > 0.05:
        return "🙂", "Slightly positive", "#99CC99"
    elif sentiment_score >= -0.05:
        return "😐", "Neutral", "#CCCCCC"
    elif sentiment_score >= -0.2:
        return "😕", "Slightly negative", "#CC9999"
    elif sentiment_score >= -0.5:
        return "😟", "Negative", "#CC6666"
    else:
        return "😡", "Very negative", "#CC0000"

def display_stock_mood(prediction_mood, sentiment_mood=None):
    """
    Display the stock mood with emojis and descriptions.
    
    Parameters:
    prediction_mood (tuple): (emoji, description, color) for prediction
    sentiment_mood (tuple, optional): (emoji, description, color) for sentiment
    
    Returns:
    None
    """
    pred_emoji, pred_desc, pred_color = prediction_mood
    
    if sentiment_mood:
        sent_emoji, sent_desc, sent_color = sentiment_mood
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: rgba(0,0,0,0.1);">
                <span style="font-size: 2em;">{pred_emoji}</span>
                <p style="margin: 0; color: {pred_color}; font-weight: bold;">Price Prediction: {pred_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[1]:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: rgba(0,0,0,0.1);">
                <span style="font-size: 2em;">{sent_emoji}</span>
                <p style="margin: 0; color: {sent_color}; font-weight: bold;">Market Sentiment: {sent_desc}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: rgba(0,0,0,0.1);">
            <span style="font-size: 2em;">{pred_emoji}</span>
            <p style="margin: 0; color: {pred_color}; font-weight: bold;">Stock Mood: {pred_desc}</p>
        </div>
        """, unsafe_allow_html=True)

def get_overall_stock_mood(prediction_mood, sentiment_mood=None):
    """
    Combine prediction and sentiment moods to get an overall stock mood.
    
    Parameters:
    prediction_mood (tuple): (emoji, description, color) for prediction
    sentiment_mood (tuple, optional): (emoji, description, color) for sentiment
    
    Returns:
    tuple: (emoji, description, color) for overall mood
    """
    if not sentiment_mood:
        return prediction_mood
    
    pred_emoji, pred_desc, pred_color = prediction_mood
    sent_emoji, sent_desc, sent_color = sentiment_mood
    
    # Map mood descriptions to numeric values
    mood_values = {
        "Very bullish": 3, "Bullish": 2, "Slightly bullish": 1, "Neutral": 0,
        "Slightly bearish": -1, "Bearish": -2, "Very bearish": -3,
        "Very positive": 3, "Positive": 2, "Slightly positive": 1,
        "Slightly negative": -1, "Negative": -2, "Very negative": -3
    }
    
    # Get numeric values for prediction and sentiment
    pred_value = mood_values.get(pred_desc, 0)
    sent_value = mood_values.get(sent_desc, 0)
    
    # Calculate combined value (weight prediction more)
    combined_value = (pred_value * 0.7) + (sent_value * 0.3)
    
    # Map combined value back to mood
    if combined_value > 2.5:
        return "🚀", "Very bullish", "#00CC00"
    elif combined_value > 1.5:
        return "😁", "Bullish", "#66CC66"
    elif combined_value > 0.5:
        return "🙂", "Slightly bullish", "#99CC99"
    elif combined_value > -0.5:
        return "😐", "Neutral", "#CCCCCC"
    elif combined_value > -1.5:
        return "😕", "Slightly bearish", "#CC9999"
    elif combined_value > -2.5:
        return "😟", "Bearish", "#CC6666"
    else:
        return "🧸", "Very bearish", "#CC0000"