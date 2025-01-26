import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # Use the Tkinter backend for interactive plots

# Fetch stock data for a given ticker (e.g., Apple - AAPL)
ticker = "AAPL"
data = yf.download(ticker, start="2022-01-01", end="2023-01-01")

# Display the first few rows of the dataset
print(data.head())
print(data.isnull().sum())
print(data.describe())

# Plot the closing prices
plt.figure(figsize=(10, 5))
plt.plot(data['Volume'], label=f"{ticker} Volume", color='orange')
plt.title(f"{ticker} Stock Volume")
plt.plot(data['Close'], label=f"{ticker} Closing Prices")
plt.title(f"{ticker} Stock Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()