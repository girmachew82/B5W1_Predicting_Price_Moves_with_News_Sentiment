import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import ta
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.set(style="whitegrid")

class QuantivativeAnalysis:
    def merge_all_stocks(self, stock_dfs: dict, date_col: str = 'Date') -> pd.DataFrame:
        """
        Merge multiple stock DataFrames into one with a 'ticker' column.

        Parameters:
        - stock_dfs (dict): Dictionary of stock tickers and their DataFrames.
        - date_col (str): Name of the date column (must exist in all DataFrames).

        Returns:
        - pd.DataFrame: Combined DataFrame of all stocks with ticker labels.
        """
        try:
            combined = []

            for ticker, df in stock_dfs.items():
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])  # Drop rows with invalid dates
                df['ticker'] = ticker
                combined.append(df)

            merged_df = pd.concat(combined, ignore_index=True)
            print(f"Combined stock data shape: {merged_df.shape}")
            return merged_df

        except Exception as e:
            print("Error in merge_all_stocks:", str(e))
            return pd.DataFrame()


    def apply_technical_indicators(self, df):
        """
        Applies common technical indicators (SMA, RSI, MACD) to the stock price DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame with stock data. Must include 'Date', 'Open', 'High', 'Low', 'Close', and 'Volume'.

        Returns:
        pd.DataFrame: Original DataFrame with added indicator columns.
        """
        try:
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Input DataFrame must contain: " + ", ".join(required_cols))

            # Convert data types
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
            df[required_cols[1:]] = df[required_cols[1:]].apply(pd.to_numeric, errors='coerce')

            # Drop rows with NaNs after type conversion
            df = df.dropna(subset=['Close'])

            # Simple Moving Average (20)
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)

            # Relative Strength Index (RSI)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

            # MACD and Signal Line
            df['MACD'] = ta.trend.macd(df['Close'])
            df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])

            return df

        except Exception as e:
            print(f"Error in apply_technical_indicators: {e}")
            return df
        

    def plot_sma(self, df, title="SMA vs Close Price"):
        import pandas_ta as ta

        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df.ta.sma(length=20)

        plt.figure(figsize=(14, 6))
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.plot(df['Date'], df['SMA_20'], label='SMA 20', color='orange')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def cast_and_merge_stocks(self, stock_dfs):
            """
            Casts the 'Date' column to datetime and merges all stock DataFrames on this column.
            
            Parameters:
            - stock_dfs (dict): A dictionary where keys are stock names and values are DataFrames.

            Returns:
            - pd.DataFrame: Merged DataFrame with suffixes for each stock's columns.
            """
            try:
                merged_df = None

                for name, df in stock_dfs.items():
                    df = df.copy()
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])  # Drop rows with invalid dates
                    df = df.sort_values('Date')

                    # Rename non-date columns with stock prefix
                    df_renamed = df.rename(columns=lambda col: f"{name}_{col}" if col != 'Date' else col)

                    if merged_df is None:
                        merged_df = df_renamed
                    else:
                        merged_df = pd.merge(merged_df, df_renamed, on='Date', how='inner')

                return merged_df

            except Exception as e:
                print(f"Error in cast_and_merge_stocks: {e}")
                return None



    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators for the stock data.

        Args:
            data (pd.DataFrame): Historical stock data

        Returns:
            pd.DataFrame: Data with added technical indicators
        """
        df = data.copy()

        # Calculate Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df