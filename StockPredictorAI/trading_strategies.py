import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st


class TradingStrategy:
    """Base class for trading strategies"""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.positions = []
        self.trades = []
        self.initial_capital = 0
        self.current_capital = 0
        self.returns = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.win_rate = 0

    def execute(self, data, predictions):
        """
        Execute the trading strategy on the given data and predictions.
        
        Parameters:
        data (pandas.DataFrame): Historical price data
        predictions (dict): Dictionary containing model predictions
        
        Returns:
        dict: Performance metrics and trade history
        """
        pass

    def _calculate_metrics(self, trade_history, prices):
        """
        Calculate performance metrics for the trading strategy.
        
        Parameters:
        trade_history (pandas.DataFrame): Trade history with dates, entry/exit prices, P&L
        prices (pandas.Series): Series of prices for the full period
        
        Returns:
        dict: Performance metrics
        """
        if trade_history.empty:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'risk_reward_ratio': 0,
                'profit_factor': 0,
                'num_trades': 0
            }

        # Calculate returns
        initial_capital = self.initial_capital
        final_capital = trade_history['cumulative_pnl'].iloc[
            -1] + initial_capital
        total_return = (final_capital / initial_capital - 1) * 100

        # Calculate annualized return
        days = (trade_history.index[-1] - trade_history.index[0]).days
        if days > 0:
            annualized_return = (
                (1 + total_return / 100)**(365 / days) - 1) * 100
        else:
            annualized_return = 0

        # Calculate max drawdown
        cumulative = trade_history['cumulative_pnl'] + initial_capital
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())

        # Calculate Sharpe ratio (simplified)
        if 'daily_returns' in trade_history.columns:
            daily_returns = trade_history['daily_returns']
            if daily_returns.std() != 0:
                sharpe_ratio = (daily_returns.mean() /
                                daily_returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Calculate win rate
        winning_trades = trade_history[trade_history['pnl'] > 0]
        win_rate = len(winning_trades) / len(trade_history) * 100 if len(
            trade_history) > 0 else 0

        # Calculate risk-reward ratio
        avg_win = winning_trades['pnl'].mean() if len(
            winning_trades) > 0 else 0
        losing_trades = trade_history[trade_history['pnl'] < 0]
        avg_loss = abs(
            losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 1
        risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0

        # Calculate profit factor
        total_profit = winning_trades['pnl'].sum() if len(
            winning_trades) > 0 else 0
        total_loss = abs(
            losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_profit / total_loss if total_loss != 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'risk_reward_ratio': risk_reward_ratio,
            'profit_factor': profit_factor,
            'num_trades': len(trade_history)
        }

    def plot_equity_curve(self, trade_history):
        """
        Plot equity curve for the trading strategy.
        
        Parameters:
        trade_history (pandas.DataFrame): Trade history with cumulative P&L
        
        Returns:
        plotly.graph_objs._figure.Figure: Plotly figure
        """
        if trade_history.empty:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=
                "No trades were executed with the current strategy and parameters.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False)
            fig.update_layout(title="Equity Curve", height=500)
            return fig

        # Create a figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Initial capital plus cumulative P&L
        equity = trade_history['cumulative_pnl'] + self.initial_capital

        # Add equity curve
        fig.add_trace(go.Scatter(x=trade_history.index,
                                 y=equity,
                                 mode='lines',
                                 name='Equity',
                                 line=dict(color='blue', width=2)),
                      secondary_y=False)

        # Add drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100

        fig.add_trace(go.Scatter(x=trade_history.index,
                                 y=drawdown,
                                 mode='lines',
                                 name='Drawdown (%)',
                                 line=dict(color='red', width=1)),
                      secondary_y=True)

        # Add buy and sell markers
        buys = trade_history[trade_history['action'] == 'BUY']
        sells = trade_history[trade_history['action'] == 'SELL']

        fig.add_trace(go.Scatter(x=buys.index,
                                 y=[equity.loc[date] for date in buys.index],
                                 mode='markers',
                                 name='Buy',
                                 marker=dict(color='green',
                                             size=8,
                                             symbol='triangle-up')),
                      secondary_y=False)

        fig.add_trace(go.Scatter(x=sells.index,
                                 y=[equity.loc[date] for date in sells.index],
                                 mode='markers',
                                 name='Sell',
                                 marker=dict(color='red',
                                             size=8,
                                             symbol='triangle-down')),
                      secondary_y=False)

        # Update layout
        fig.update_layout(title=f"Equity Curve - {self.name}",
                          xaxis_title="Date",
                          yaxis_title="Equity ($)",
                          legend_title="Legend",
                          height=500)

        # Update y-axis labels
        fig.update_yaxes(title_text="Equity ($)", secondary_y=False)
        fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True)

        return fig

    def plot_trade_distribution(self, trade_history):
        """
        Plot trade distribution for the trading strategy.
        
        Parameters:
        trade_history (pandas.DataFrame): Trade history with P&L
        
        Returns:
        plotly.graph_objs._figure.Figure: Plotly figure
        """
        if trade_history.empty:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=
                "No trades were executed with the current strategy and parameters.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False)
            fig.update_layout(title="Trade Distribution", height=400)
            return fig

        # Create subplots
        fig = make_subplots(rows=1,
                            cols=2,
                            subplot_titles=("P&L Distribution",
                                            "Trade Duration"))

        # Add P&L histogram
        fig.add_trace(go.Histogram(x=trade_history['pnl'],
                                   name='P&L',
                                   marker_color=[
                                       'green' if x > 0 else 'red'
                                       for x in trade_history['pnl']
                                   ],
                                   opacity=0.7,
                                   nbinsx=20),
                      row=1,
                      col=1)

        # Add trade duration histogram
        if 'duration_days' in trade_history.columns:
            fig.add_trace(go.Histogram(x=trade_history['duration_days'],
                                       name='Duration (days)',
                                       marker_color='blue',
                                       opacity=0.7,
                                       nbinsx=20),
                          row=1,
                          col=2)

        # Update layout
        fig.update_layout(title=f"Trade Distribution - {self.name}",
                          height=400,
                          showlegend=False)

        # Update x-axis labels
        fig.update_xaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (days)", row=1, col=2)

        return fig


class SimpleMovingAverageCrossover(TradingStrategy):
    """
    Simple Moving Average Crossover strategy.
    Goes long when fast MA crosses above slow MA and short when fast MA crosses below slow MA.
    """

    def __init__(self, fast_period=20, slow_period=50, initial_capital=10000):
        super().__init__(
            name="SMA Crossover",
            description=
            f"Moving Average Crossover ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.initial_capital = initial_capital

    def execute(self, data, predictions=None, use_predictions=False):
        """
        Execute the SMA Crossover strategy.
        
        Parameters:
        data (pandas.DataFrame): Historical price data
        predictions (dict, optional): Dictionary containing model predictions
        use_predictions (bool): Whether to use model predictions to adjust signals
        
        Returns:
        dict: Performance metrics and trade history
        """
        # Make a copy of the data
        df = data.copy()

        # Calculate moving averages
        df['fast_ma'] = df['Close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['Close'].rolling(window=self.slow_period).mean()

        # Calculate crossover signals
        df['signal'] = 0
        df['position'] = 0

        # Crossover signals
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # Buy signal
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Sell signal

        # Filter out initial NaN values
        df = df.dropna()

        # Calculate position changes
        df['position'] = df['signal']
        position_changes = df['position'].diff()

        # Incorporate predictions if available and requested
        if use_predictions and predictions is not None:
            # Use an average of model predictions if available
            pred_models = list(predictions.keys())
            if pred_models:
                # Get the last date in our historical data
                last_date = df.index[-1]

                # Calculate average predicted direction for next day
                next_day_change = 0
                for model in pred_models:
                    if len(predictions[model]) > 0:
                        # Compare first prediction to last actual price
                        pred_direction = 1 if predictions[model][0] > df[
                            'Close'].iloc[-1] else -1
                        next_day_change += pred_direction

                avg_direction = np.sign(next_day_change)

                # Adjust the last signal based on predictions
                if avg_direction != 0:
                    df.loc[df.index[-1], 'signal'] = avg_direction
                    df.loc[df.index[-1], 'position'] = avg_direction
                    position_changes.iloc[
                        -1] = df['position'].iloc[-1] - df['position'].iloc[-2]

        # Initialize trade tracking
        trades = []
        current_position = 0
        entry_price = 0
        entry_date = None
        cash = self.initial_capital
        equity = self.initial_capital
        holdings = 0
        trade_history = []

        # Track daily changes
        daily_pnl = []
        cumulative_pnl = 0

        # Iterate through the data
        for date, row in df.iterrows():
            price = row['Close']
            signal = row['position']
            pos_change = position_changes.loc[
                date] if date in position_changes.index else 0

            # Check for position changes
            if pos_change != 0:
                # Position size - here we use all available capital for simplicity
                # In a real strategy, you would use proper position sizing
                position_size = cash // price if cash >= price else 0

                if pos_change > 0:  # Buy signal
                    # Record entry
                    if position_size > 0:
                        current_position = position_size
                        entry_price = price
                        entry_date = date

                        # Update cash and holdings
                        cash -= position_size * price
                        holdings = position_size

                        # Add to trade history
                        trade_history.append({
                            'date': date,
                            'action': 'BUY',
                            'price': price,
                            'quantity': position_size,
                            'value': position_size * price,
                            'cash': cash,
                            'holdings': holdings,
                            'equity': cash + holdings * price,
                            'pnl': 0,
                            'cumulative_pnl': cumulative_pnl
                        })

                elif pos_change < 0 and current_position > 0:  # Sell signal with existing position
                    # Calculate P&L
                    exit_price = price
                    pnl = (exit_price - entry_price) * current_position
                    cumulative_pnl += pnl

                    # Update cash and holdings
                    cash += current_position * exit_price
                    holdings = 0

                    # Record trade
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': current_position,
                        'pnl': pnl,
                        'return': (exit_price / entry_price - 1) * 100,
                        'duration_days': (date - entry_date).days
                    })

                    # Add to trade history
                    trade_history.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'quantity': current_position,
                        'value': current_position * price,
                        'cash': cash,
                        'holdings': holdings,
                        'equity': cash + holdings * price,
                        'pnl': pnl,
                        'cumulative_pnl': cumulative_pnl
                    })

                    # Reset position tracking
                    current_position = 0
                    entry_price = 0
                    entry_date = None

            # Calculate daily equity
            equity = cash + holdings * price

            # Calculate daily return
            if len(daily_pnl) > 0:
                daily_return = (equity / daily_pnl[-1]['equity']) - 1
            else:
                daily_return = 0

            # Add to daily P&L tracking
            daily_pnl.append({
                'date': date,
                'equity': equity,
                'daily_return': daily_return
            })

        # Process results
        if not trade_history:
            # Create empty dataframe with proper columns if no trades
            trade_df = pd.DataFrame(columns=[
                'date', 'action', 'price', 'quantity', 'value', 'cash',
                'holdings', 'equity', 'pnl', 'cumulative_pnl'
            ])
            trade_df.set_index('date', inplace=True)

            # Return empty results
            return {
                'metrics': self._calculate_metrics(trade_df, df['Close']),
                'trades': pd.DataFrame(trades),
                'trade_history': trade_df,
                'equity_curve': self.plot_equity_curve(trade_df),
                'trade_distribution': self.plot_trade_distribution(trade_df),
                'position': 0,
                'cash': self.initial_capital,
                'holdings': 0
            }

        # Create trade history dataframe
        trade_df = pd.DataFrame(trade_history)
        trade_df.set_index('date', inplace=True)

        # Create daily returns dataframe
        daily_df = pd.DataFrame(daily_pnl)
        daily_df.set_index('date', inplace=True)

        # Add daily returns to trade history
        trade_df['daily_returns'] = daily_df['daily_return']

        # Calculate performance metrics
        metrics = self._calculate_metrics(trade_df, df['Close'])

        # Store results
        self.trades = trades
        self.metrics = metrics

        # Create trade dataframe
        trades_df = pd.DataFrame(trades)

        return {
            'metrics': metrics,
            'trades': trades_df,
            'trade_history': trade_df,
            'equity_curve': self.plot_equity_curve(trade_df),
            'trade_distribution': self.plot_trade_distribution(trades_df),
            'position': current_position,
            'cash': cash,
            'holdings': holdings
        }


class PredictionBasedStrategy(TradingStrategy):
    """
    A strategy that makes trading decisions based on machine learning model predictions.
    """

    def __init__(self,
                 threshold=0.5,
                 holding_period=5,
                 initial_capital=10000,
                 model_weights=None):
        super().__init__(
            name="ML Prediction Strategy",
            description=
            f"Trades based on ML predictions with {threshold}% threshold")
        self.threshold = threshold
        self.holding_period = holding_period
        self.initial_capital = initial_capital
        self.model_weights = model_weights or {
            'Linear Regression': 0.25,
            'Ridge': 0.25,
            'Random Forest': 0.25,
            'Gradient Boosting': 0.25
        }

    def execute(self, data, predictions, prediction_dates=None):
        """
        Execute the ML Prediction strategy.
        
        Parameters:
        data (pandas.DataFrame): Historical price data
        predictions (dict): Dictionary containing model predictions
        prediction_dates (list, optional): List of dates for predictions
        
        Returns:
        dict: Performance metrics and trade history
        """
        # Make a copy of the data
        df = data.copy()

        # Ensure df is sorted by date
        df = df.sort_index()

        # We'll simulate trading on the test set
        prediction_df = pd.DataFrame(index=df.index)
        prediction_df['Close'] = df['Close']

        # Initialize signals
        prediction_df['signal'] = 0

        # Calculate weighted average predictions
        if prediction_dates is None:
            # Use historical data dates
            if len(predictions) > 0 and isinstance(
                    list(predictions.values())[0], np.ndarray):
                # Get predictions for each model
                for model, preds in predictions.items():
                    if model in self.model_weights:
                        weight = self.model_weights[model]
                        pred_len = min(len(preds), len(prediction_df))
                        prediction_df.iloc[
                            -pred_len:,
                            prediction_df.columns.get_indexer(
                                ['signal'])] += preds[:pred_len] * weight
        else:
            # Future predictions - combine into a dataframe
            future_df = pd.DataFrame(index=prediction_dates)

            # Get weighted average predictions
            future_df['signal'] = 0

            for model, preds in predictions.items():
                if model in self.model_weights:
                    weight = self.model_weights[model]
                    pred_len = min(len(preds), len(future_df))
                    future_preds = np.array(preds[:pred_len])
                    future_df.iloc[:pred_len,
                                   future_df.columns.get_indexer(
                                       ['signal'])] += future_preds * weight

            # Calculate percent change from last known price
            last_price = df['Close'].iloc[-1]
            future_df['predicted_price'] = future_df['signal']
            future_df['pct_change'] = (
                future_df['predicted_price'] / last_price - 1) * 100

            # Determine signals based on threshold
            future_df.loc[future_df['pct_change'] > self.threshold,
                          'signal'] = 1
            future_df.loc[future_df['pct_change'] < -self.threshold,
                          'signal'] = -1

            return {
                'future_signals': future_df,
                'last_position': 0,
                'last_price': last_price
            }

        # Calculate price changes to compare with predictions
        prediction_df['pct_change'] = prediction_df['Close'].pct_change() * 100

        # Set signals based on threshold
        prediction_df.loc[prediction_df['pct_change'] > self.threshold,
                          'signal'] = 1
        prediction_df.loc[prediction_df['pct_change'] < -self.threshold,
                          'signal'] = -1

        # Drop rows with NaN
        prediction_df = prediction_df.dropna()

        # Calculate position changes
        prediction_df['position'] = prediction_df['signal']
        position_changes = prediction_df['position'].diff()

        # Initialize trade tracking
        trades = []
        current_position = 0
        entry_price = 0
        entry_date = None
        cash = self.initial_capital
        equity = self.initial_capital
        holdings = 0
        trade_history = []

        # Track daily changes
        daily_pnl = []
        cumulative_pnl = 0

        # Iterate through the data
        for date, row in prediction_df.iterrows():
            price = row['Close']
            signal = row['position']
            pos_change = position_changes.loc[
                date] if date in position_changes.index else 0

            # Check for position changes
            if pos_change != 0:
                # Position size - here we use 90% of available capital
                position_size = int(
                    (cash * 0.9) / price) if cash >= price else 0

                if pos_change > 0:  # Buy signal
                    # Record entry
                    if position_size > 0:
                        current_position = position_size
                        entry_price = price
                        entry_date = date

                        # Update cash and holdings
                        cash -= position_size * price
                        holdings = position_size

                        # Add to trade history
                        trade_history.append({
                            'date': date,
                            'action': 'BUY',
                            'price': price,
                            'quantity': position_size,
                            'value': position_size * price,
                            'cash': cash,
                            'holdings': holdings,
                            'equity': cash + holdings * price,
                            'pnl': 0,
                            'cumulative_pnl': cumulative_pnl
                        })

                elif pos_change < 0 and current_position > 0:  # Sell signal with existing position
                    # Calculate P&L
                    exit_price = price
                    pnl = (exit_price - entry_price) * current_position
                    cumulative_pnl += pnl

                    # Update cash and holdings
                    cash += current_position * exit_price
                    holdings = 0

                    # Record trade
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': current_position,
                        'pnl': pnl,
                        'return': (exit_price / entry_price - 1) * 100,
                        'duration_days': (date - entry_date).days
                    })

                    # Add to trade history
                    trade_history.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'quantity': current_position,
                        'value': current_position * price,
                        'cash': cash,
                        'holdings': holdings,
                        'equity': cash + holdings * price,
                        'pnl': pnl,
                        'cumulative_pnl': cumulative_pnl
                    })

                    # Reset position tracking
                    current_position = 0
                    entry_price = 0
                    entry_date = None

            # Calculate daily equity
            equity = cash + holdings * price

            # Calculate daily return
            if len(daily_pnl) > 0:
                daily_return = (equity / daily_pnl[-1]['equity']) - 1
            else:
                daily_return = 0

            # Add to daily P&L tracking
            daily_pnl.append({
                'date': date,
                'equity': equity,
                'daily_return': daily_return
            })

        # Process results
        if not trade_history:
            # Create empty dataframe with proper columns if no trades
            trade_df = pd.DataFrame(columns=[
                'date', 'action', 'price', 'quantity', 'value', 'cash',
                'holdings', 'equity', 'pnl', 'cumulative_pnl'
            ])
            trade_df.set_index('date', inplace=True)

            # Return empty results
            return {
                'metrics': self._calculate_metrics(trade_df, df['Close']),
                'trades': pd.DataFrame(trades),
                'trade_history': trade_df,
                'equity_curve': self.plot_equity_curve(trade_df),
                'trade_distribution': self.plot_trade_distribution(trade_df),
                'position': 0,
                'cash': self.initial_capital,
                'holdings': 0
            }

        # Create trade history dataframe
        trade_df = pd.DataFrame(trade_history)
        trade_df.set_index('date', inplace=True)

        # Create daily returns dataframe
        daily_df = pd.DataFrame(daily_pnl)
        daily_df.set_index('date', inplace=True)

        # Add daily returns to trade history
        trade_df['daily_returns'] = daily_df['daily_return']

        # Calculate performance metrics
        metrics = self._calculate_metrics(trade_df, df['Close'])

        # Store results
        self.trades = trades
        self.metrics = metrics

        # Create trade dataframe
        trades_df = pd.DataFrame(trades)

        return {
            'metrics': metrics,
            'trades': trades_df,
            'trade_history': trade_df,
            'equity_curve': self.plot_equity_curve(trade_df),
            'trade_distribution': self.plot_trade_distribution(trades_df),
            'position': current_position,
            'cash': cash,
            'holdings': holdings
        }


class RSIMeanReversionStrategy(TradingStrategy):
    """
    RSI Mean Reversion strategy.
    Buy when RSI is below oversold level and sell when RSI is above overbought level.
    """

    def __init__(self,
                 rsi_period=14,
                 oversold=30,
                 overbought=70,
                 initial_capital=10000,
                 use_sentiment=False):
        super().__init__(
            name="RSI Mean Reversion",
            description=f"RSI({rsi_period}) Mean Reversion Strategy")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.initial_capital = initial_capital
        self.use_sentiment = use_sentiment

    def execute(self, data, sentiment_data=None):
        """
        Execute the RSI Mean Reversion strategy.
        
        Parameters:
        data (pandas.DataFrame): Historical price data
        sentiment_data (dict, optional): Dictionary containing sentiment analysis results
        
        Returns:
        dict: Performance metrics and trade history
        """
        # Make a copy of the data
        df = data.copy()

        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Initialize signals
        df['signal'] = 0

        # Generate signals based on RSI
        df.loc[df['RSI'] < self.oversold, 'signal'] = 1  # Buy signal
        df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell signal

        # Incorporate sentiment if requested
        if self.use_sentiment and sentiment_data is not None:
            # Get sentiment scores from sentiment data
            sentiment_scores = {}

            # Extract sentiment scores for dates that match our data
            for date in df.index:
                date_str = date.strftime('%Y-%m-%d')
                if date_str in sentiment_data:
                    sentiment_scores[date] = sentiment_data[date_str][
                        'overall_sentiment']

            # Adjust signals based on sentiment
            for date, sentiment in sentiment_scores.items():
                if date in df.index:
                    # Strengthen buy signals if sentiment is positive
                    if sentiment > 0.2 and df.loc[date, 'signal'] == 1:
                        df.loc[date, 'signal'] = 2  # Stronger buy

                    # Strengthen sell signals if sentiment is negative
                    elif sentiment < -0.2 and df.loc[date, 'signal'] == -1:
                        df.loc[date, 'signal'] = -2  # Stronger sell

                    # Potentially override signals based on strong sentiment
                    elif sentiment > 0.5 and df.loc[date, 'signal'] == 0:
                        df.loc[date, 'signal'] = 1  # Generate buy

                    elif sentiment < -0.5 and df.loc[date, 'signal'] == 0:
                        df.loc[date, 'signal'] = -1  # Generate sell

        # Drop rows with NaN
        df = df.dropna()

        # Calculate position
        df['position'] = df['signal'].replace(2, 1).replace(
            -2, -1)  # Normalize to -1, 0, 1
        position_changes = df['position'].diff()

        # Initialize trade tracking
        trades = []
        current_position = 0
        entry_price = 0
        entry_date = None
        cash = self.initial_capital
        equity = self.initial_capital
        holdings = 0
        trade_history = []

        # Track daily changes
        daily_pnl = []
        cumulative_pnl = 0

        # Iterate through the data
        for date, row in df.iterrows():
            price = row['Close']
            signal = row['position']
            pos_change = position_changes.loc[
                date] if date in position_changes.index else 0

            # Check for position changes
            if pos_change != 0:
                # Position size - use different sizing based on signal strength
                signal_strength = 1
                if self.use_sentiment and sentiment_data is not None:
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str in sentiment_data:
                        sentiment = sentiment_data[date_str][
                            'overall_sentiment']
                        # Adjust position size based on sentiment strength
                        if abs(sentiment) > 0.5:
                            signal_strength = 1.2  # Increase position by 20%
                        elif abs(sentiment) > 0.2:
                            signal_strength = 1.1  # Increase position by 10%

                position_size = int((cash * 0.9 * signal_strength) /
                                    price) if cash >= price else 0

                if pos_change > 0:  # Buy signal
                    # Record entry
                    if position_size > 0:
                        current_position = position_size
                        entry_price = price
                        entry_date = date

                        # Update cash and holdings
                        cash -= position_size * price
                        holdings = position_size

                        # Add to trade history
                        trade_history.append({
                            'date': date,
                            'action': 'BUY',
                            'price': price,
                            'quantity': position_size,
                            'value': position_size * price,
                            'cash': cash,
                            'holdings': holdings,
                            'equity': cash + holdings * price,
                            'pnl': 0,
                            'cumulative_pnl': cumulative_pnl
                        })

                elif pos_change < 0 and current_position > 0:  # Sell signal with existing position
                    # Calculate P&L
                    exit_price = price
                    pnl = (exit_price - entry_price) * current_position
                    cumulative_pnl += pnl

                    # Update cash and holdings
                    cash += current_position * exit_price
                    holdings = 0

                    # Record trade
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': current_position,
                        'pnl': pnl,
                        'return': (exit_price / entry_price - 1) * 100,
                        'duration_days': (date - entry_date).days
                    })

                    # Add to trade history
                    trade_history.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'quantity': current_position,
                        'value': current_position * price,
                        'cash': cash,
                        'holdings': holdings,
                        'equity': cash + holdings * price,
                        'pnl': pnl,
                        'cumulative_pnl': cumulative_pnl
                    })

                    # Reset position tracking
                    current_position = 0
                    entry_price = 0
                    entry_date = None

            # Calculate daily equity
            equity = cash + holdings * price

            # Calculate daily return
            if len(daily_pnl) > 0:
                daily_return = (equity / daily_pnl[-1]['equity']) - 1
            else:
                daily_return = 0

            # Add to daily P&L tracking
            daily_pnl.append({
                'date': date,
                'equity': equity,
                'daily_return': daily_return
            })

        # Process results
        if not trade_history:
            # Create empty dataframe with proper columns if no trades
            trade_df = pd.DataFrame(columns=[
                'date', 'action', 'price', 'quantity', 'value', 'cash',
                'holdings', 'equity', 'pnl', 'cumulative_pnl'
            ])
            trade_df.set_index('date', inplace=True)

            # Return empty results
            return {
                'metrics': self._calculate_metrics(trade_df, df['Close']),
                'trades': pd.DataFrame(trades),
                'trade_history': trade_df,
                'equity_curve': self.plot_equity_curve(trade_df),
                'trade_distribution': self.plot_trade_distribution(trade_df),
                'position': 0,
                'cash': self.initial_capital,
                'holdings': 0
            }

        # Create trade history dataframe
        trade_df = pd.DataFrame(trade_history)
        trade_df.set_index('date', inplace=True)

        # Create daily returns dataframe
        daily_df = pd.DataFrame(daily_pnl)
        daily_df.set_index('date', inplace=True)

        # Add daily returns to trade history
        trade_df['daily_returns'] = daily_df['daily_return']

        # Calculate performance metrics
        metrics = self._calculate_metrics(trade_df, df['Close'])

        # Store results
        self.trades = trades
        self.metrics = metrics

        # Create trade dataframe
        trades_df = pd.DataFrame(trades)

        return {
            'metrics': metrics,
            'trades': trades_df,
            'trade_history': trade_df,
            'equity_curve': self.plot_equity_curve(trade_df),
            'trade_distribution': self.plot_trade_distribution(trades_df),
            'position': current_position,
            'cash': cash,
            'holdings': holdings
        }


def get_strategy_by_name(strategy_name, params=None):
    """
    Get a trading strategy instance by name.
    
    Parameters:
    strategy_name (str): Name of the strategy
    params (dict): Dictionary of parameters for the strategy
    
    Returns:
    TradingStrategy: Instance of the requested strategy
    """
    params = params or {}

    strategy_map = {
        'SMA Crossover': SimpleMovingAverageCrossover,
        'ML Prediction': PredictionBasedStrategy,
        'RSI Mean Reversion': RSIMeanReversionStrategy
    }

    if strategy_name in strategy_map:
        return strategy_map[strategy_name](**params)
    else:
        raise ValueError(
            f"Strategy '{strategy_name}' not found. Available strategies: {list(strategy_map.keys())}"
        )


def get_available_strategies():
    """
    Get a list of available trading strategies.
    
    Returns:
    list: List of strategy names
    """
    return ['SMA Crossover', 'ML Prediction', 'RSI Mean Reversion']


def generate_trading_plan(strategy_results, asset_name):
    """
    Generate a trading plan based on strategy results.
    
    Parameters:
    strategy_results (dict): Dictionary containing strategy execution results
    asset_name (str): Name of the asset
    
    Returns:
    str: Trading plan text
    """
    metrics = strategy_results.get('metrics', {})
    position = strategy_results.get('position', 0)

    trades = strategy_results.get('trades', pd.DataFrame())

    # Current position and recommendation
    current_position_text = "No position" if position == 0 else f"Long {position} shares"

    # Analyze recent performance
    avg_return = trades['return'].mean() if not trades.empty else 0
    win_rate = metrics.get('win_rate', 0)

    # Trading recommendations
    if position > 0:
        recommendation = "Hold current position and monitor for exit signals."
    elif avg_return > 2 and win_rate > 50:
        recommendation = "Look for entry opportunities following the strategy signals."
    else:
        recommendation = "Remain cautious and wait for stronger signals before entering."

    # Risk management
    if not trades.empty and 'pnl' in trades.columns:
        avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if len(
            trades[trades['pnl'] < 0]) > 0 else 0
        suggested_stop = f"Consider setting stop losses at approximately ${avg_loss:.2f} below entry price."
    else:
        suggested_stop = "Set appropriate stop losses based on your risk tolerance."

    # Generate the trading plan text
    trading_plan = f"""
    # Trading Plan for {asset_name}
    
    ## Current Status
    - **Current Position**: {current_position_text}
    - **Strategy Performance**: {win_rate:.1f}% win rate, {avg_return:.2f}% average return per trade
    
    ## Recommendation
    {recommendation}
    
    ## Risk Management
    - {suggested_stop}
    - Position sizing: Limit each position to 5-10% of portfolio value.
    
    ## Trading Rules
    1. Follow the strategy signals strictly.
    2. Document each trade and the reason for entry/exit.
    3. Review the strategy performance regularly.
    4. Adjust parameters if needed based on changing market conditions.
    
    ## Next Steps
    - Set price alerts based on strategy parameters.
    - Pre-define exit conditions for both profit taking and stop losses.
    - Monitor overall market conditions that might affect this asset.
    """

    return trading_plan
