import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

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
                    df['ticker'] = ticker
                    combined.append(df)

                merged_df = pd.concat(combined, ignore_index=True)
                print(f"Combined stock data shape: {merged_df.shape}")
                return merged_df

            except Exception as e:
                print("Error in merge_all_stocks:", str(e))
                return pd.DataFrame()


