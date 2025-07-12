import os
import requests
import pandas as pd
from dotenv import load_dotenv

def fetch_and_save_stock_data(ticker, start, end, filename="data/prices/stock_prices.csv"):
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("❌ ALPHAVANTAGE_API_KEY not found.")
        return

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}&datatype=csv"

    try:
        df = pd.read_csv(url)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        df = df.rename(columns={'timestamp': 'Date', 'adjusted_close': 'Close'})
        df['ticker'] = ticker
        df.to_csv(filename, index=False)
        print(f"✅ Saved stock data for {ticker} to {filename}")
    except Exception as e:
        print(f"❌ Error fetching price data: {e}")