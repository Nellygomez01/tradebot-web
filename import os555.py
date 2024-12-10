import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
import requests
from textblob import TextBlob
import time
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
import yfinance as yf
import ta
import warnings
warnings.filterwarnings('ignore')

class EnhancedCryptoTradingBot:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': 'zfwyJlkyD32CZjaf0G',
            'secret': 'lirMdi0jPr4ynPdUcBwkznnNvBW7lHogtKh8'
        })
        self.models = {
            'lstm': self._build_lstm_model(),
            'gru': self._build_gru_model(),
            'rf': RandomForestClassifier(n_estimators=100)
        }
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.scaler = MinMaxScaler()
        self.trading_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'BNB/USD']
        
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        return model

    def _build_gru_model(self):
        model = Sequential([
            GRU(100, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        return model

    def fetch_comprehensive_data(self, symbol, timeframe='1h', limit=1000):
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add technical indicators
            df = self.add_advanced_indicators(df)
            
            # Add market depth
            order_book = self.exchange.fetch_order_book(symbol)
            df['bid_ask_spread'] = pd.Series([order_book['asks'][0][0] - order_book['bids'][0][0]])
            
            return df
        except Exception as e:
            print(f"Error fetching comprehensive data: {e}")
            return None

    def add_advanced_indicators(self, df):
        # Volume-based indicators
        df['volume_sma'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        
        # Trend indicators
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'])
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = ta.volatility.bollinger_bands(df['close'])
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        return df

    def get_enhanced_sentiment(self):
        try:
            # Crypto news sentiment
            crypto_news = requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN").json()['Data']
            news_sentiment = []
            
            for news in crypto_news[:20]:
                sentiment = self.sentiment_analyzer(news['title'])[0]
                score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
                news_sentiment.append(score)
            
            # Social media sentiment (placeholder - would need actual API integration)
            social_sentiment = 0  # Implement social media sentiment analysis
            
            # Market sentiment indicators
            fear_greed = requests.get("https://api.alternative.me/fng/").json()['data'][0]['value']
            
            # Combine all sentiment sources
            combined_sentiment = (np.mean(news_sentiment) + social_sentiment + int(fear_greed)/100) / 3
            return combined_sentiment
            
        except Exception as e:
            print(f"Error in enhanced sentiment analysis: {e}")
            return 0

class AITradingAgent:
    def __init__(self, bot):
        self.bot = bot
        self.confidence_threshold = 0.8
        self.max_position_size = 0.05  # 5% of portfolio
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.portfolio_value = self.get_portfolio_value()

    def get_portfolio_value(self):
        try:
            balance = self.bot.exchange.fetch_balance()
            return balance['total']['USD']
        except:
            return 10000  # Default value

    def analyze_trading_opportunity(self, symbol):
        df = self.bot.fetch_comprehensive_data(symbol)
        if df is None:
            return None

        sentiment = self.bot.get_enhanced_sentiment()
        
        # Multi-model prediction ensemble
        lstm_pred = self.bot.models['lstm'].predict(self.prepare_data(df['close']))
        gru_pred = self.bot.models['gru'].predict(self.prepare_data(df['close']))
        
        # Combine predictions with weights
        ensemble_prediction = (lstm_pred * 0.4 + gru_pred * 0.6)[0][0]
        
        # Calculate position size based on Kelly Criterion
        win_rate = self.calculate_win_rate(df)
        kelly_fraction = win_rate - ((1 - win_rate) / (self.max_risk_per_trade / 0.02))
        position_size = min(kelly_fraction * self.portfolio_value, self.max_position_size * self.portfolio_value)

        return self.generate_trade_signal(df, sentiment, ensemble_prediction, position_size)

    def prepare_data(self, data):
        scaled_data = self.bot.scaler.fit_transform(data.values.reshape(-1, 1))
        sequences = []
        for i in range(len(scaled_data) - 60):
            sequences.append(scaled_data[i:(i + 60)])
        return np.array(sequences)

    def calculate_win_rate(self, df):
        # Implement win rate calculation based on historical performance
        return 0.6  # Placeholder

    def generate_trade_signal(self, df, sentiment, prediction, position_size):
        current_price = df['close'].iloc[-1]
        
        # Advanced signal generation logic
        signal = None
        confidence = 0
        
        # Long signal conditions
        if (prediction > current_price * 1.02 and  # 2% predicted increase
            sentiment > 0.2 and
            df['rsi'].iloc[-1] < 30 and
            df['cci'].iloc[-1] < -100):
            
            signal = 'buy'
            confidence = 0.85
            
        # Short signal conditions    
        elif (prediction < current_price * 0.98 and  # 2% predicted decrease
              sentiment < -0.2 and
              df['rsi'].iloc[-1] > 70 and
              df['cci'].iloc[-1] > 100):
              
            signal = 'sell'
            confidence = 0.85

        if confidence >= self.confidence_threshold:
            return {
                'signal': signal,
                'position_size': position_size,
                'stop_loss': current_price * (0.95 if signal == 'buy' else 1.05),
                'take_profit': current_price * (1.1 if signal == 'buy' else 0.9),
                'confidence': confidence
            }
            
        return None

# Initialize and run enhanced trading system
if __name__ == "__main__":
    bot = EnhancedCryptoTradingBot()
    agent = AITradingAgent(bot)
    
    while True:
        try:
            for symbol in bot.trading_pairs:
                trade_info = agent.analyze_trading_opportunity(symbol)
                if trade_info:
                    order = bot.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=trade_info['signal'],
                        amount=trade_info['position_size'],
                        params={
                            'stopLoss': trade_info['stop_loss'],
                            'takeProfit': trade_info['take_profit']
                        }
                    )
                    print(f"Executed {trade_info['signal']} order for {symbol}: {order}")
                    print(f"Confidence: {trade_info['confidence']}")
                    
            time.sleep(300)  # 5-minute interval
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)
