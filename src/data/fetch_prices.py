# Folder: src/data/fetch_prices.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

def fetch_and_save_stock_data(ticker, start, end, filename="data/prices/stock_prices.csv"):
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("❌ ALPHAVANTAGE_API_KEY not found.")
        return

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}&datatype=csv"

    try:
        df_new = pd.read_csv(url)
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
        df_new = df_new[(df_new['timestamp'] >= start) & (df_new['timestamp'] <= end)]
        df_new = df_new.rename(columns={'timestamp': 'Date', 'adjusted_close': 'Close'})
        df_new['ticker'] = ticker
        df_new['start_date'] = start
        df_new['end_date'] = end
        df_new['fetch_date'] = pd.Timestamp.now().date()

        try:
            df_existing = pd.read_csv(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['ticker', 'Date'], inplace=True)
        except FileNotFoundError:
            df_combined = df_new

        df_combined['Date'] = pd.to_datetime(df_combined['Date'], errors='coerce')
        df_combined.sort_values(by=['ticker', 'Date'], inplace=True)
        df_combined.to_csv(filename, index=False)
        print(f"✅ Saved stock data for {ticker} to '{filename}' with {len(df_combined)} rows.")
    except Exception as e:
        print(f"❌ Error fetching price data: {e}")