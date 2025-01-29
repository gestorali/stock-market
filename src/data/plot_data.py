import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend for interactive plots
import matplotlib.pyplot as plt

def plot_stock_data(data, ticker):
    """
    Plots the stock data for closing prices and volume.

    Parameters:
        data (pd.DataFrame): The stock data containing 'Close' and 'Volume' columns.
        ticker (str): The ticker symbol for the stock.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot closing prices on the primary y-axis
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)", color="tab:blue")
    ax1.plot(data.index, data['Close'], label=f"{ticker} Closing Prices", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # Plot volume on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Volume", color="tab:orange")
    ax2.plot(data.index, data['Volume'], label=f"{ticker} Volume", color="tab:orange", alpha=0.6)
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    # Add title and grid
    plt.title(f"{ticker} Stock Data")
    plt.grid()
    plt.show()