# Folder: src/data/fetch_prices.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

def fetch_and_save_stock_data(ticker, start, end, filename="data/prices/stock_prices.csv"):
    load_dotenv()
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("❌ FINNHUB_API_KEY not found.")
        return

    # Convert dates to UNIX timestamps (required by Finnhub)
    from_unix = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    to_unix = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

    url = (
        f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D"
        f"&from={from_unix}&to={to_unix}&token={api_key}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("s") != "ok":
            print(f"❌ No data returned for {ticker}. API response: {data}")
            return

        df_new = pd.DataFrame({
            "Date": pd.to_datetime(data["t"], unit='s'),
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"]
        })

        df_new['ticker'] = ticker
        df_new['start_date'] = start
        df_new['end_date'] = end
        df_new['fetch_date'] = pd.Timestamp.now().date()

        # Append to existing file if exists
        try:
            df_existing = pd.read_csv(filename)
            if 'Date' in df_existing.columns:
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
