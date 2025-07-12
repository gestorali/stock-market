# File: src/data/fetch_prices.py
import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_and_save_stock_data(ticker, start, end, filename="data/prices/stock_prices.csv"):
    df_new = yf.download(ticker, start=start, end=end)

    if df_new.empty:
        print(f"❌ No stock data found for {ticker}.")
        return

    df_new = df_new.reset_index()  # ensure 'Date' is a column
    df_new["ticker"] = ticker
    df_new["start_date"] = start
    df_new["end_date"] = end
    df_new["fetch_date"] = pd.Timestamp.now().date()

    # Optional: standardize column names if needed
    df_new.columns = [col.strip() for col in df_new.columns]

    try:
        df_existing = pd.read_csv(filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["ticker", "Date"], inplace=True)
    except FileNotFoundError:
        df_combined = df_new

    # Sort by date
    df_combined["Date"] = pd.to_datetime(df_combined["Date"], errors="coerce")
    df_combined.sort_values(by=["ticker", "Date"], inplace=True)

    df_combined.to_csv(filename, index=False)
    print(f"✅ Saved stock data for {ticker} to '{filename}' with {len(df_combined)} rows.")
