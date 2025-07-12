import yfinance as yf
import os

def fetch_and_save_stock_data(ticker, start, end, filename="data/prices/stock_prices.csv"):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        print("❌ No stock data found.")
        return
    df = df.reset_index()
    df["ticker"] = ticker
    df.to_csv(filename, index=False)
    print(f"✅ Saved stock data for {ticker} to {filename}")