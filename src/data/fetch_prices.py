# Folder: src/data/fetch_prices.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv
# from datetime import datetime

def fetch_and_save_stock_data(ticker, start, end, filename="data/prices/stock_prices.csv"):
    load_dotenv()
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        print("❌ TWELVE_DATA_API_KEY not found.")
        return

    url = (
        f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day"
        f"&start_date={start}&end_date={end}&apikey={api_key}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "values" not in data:
            print(f"❌ API error: {data.get('message', 'Unknown error')}")
            return

        df_new = pd.DataFrame(data["values"])
        df_new['datetime'] = pd.to_datetime(df_new['datetime'])
        df_new = df_new.rename(columns={
            'datetime': 'Date',
            'close': 'Close'
        })
        df_new['Close'] = pd.to_numeric(df_new['Close'], errors='coerce')
        df_new['ticker'] = ticker
        df_new['start_date'] = start
        df_new['end_date'] = end
        df_new['fetch_date'] = pd.Timestamp.now().date()

        try:
            df_existing = pd.read_csv(filename)
            df_existing['Date'] = pd.to_datetime(df_existing['Date'], errors='coerce')
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['ticker', 'Date'], inplace=True)
        except FileNotFoundError:
            df_combined = df_new

        df_combined.sort_values(by=['ticker', 'Date'], inplace=True)
        df_combined.to_csv(filename, index=False)
        print(f"✅ Saved stock data for {ticker} to '{filename}' with {len(df_combined)} rows.")
    except Exception as e:
        print(f"❌ Error fetching price data: {e}")
