from src.data.fetch_data import fetch_stock_data
from src.data.plot_data import plot_stock_data
from src.data.feature_engineering import add_features

def main():
    # Define parameters
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    # Step 1: Fetch data
    print("Fetching stock data...")
    data = fetch_stock_data(ticker, start_date, end_date)
    print("Data fetched successfully!")
    print(data.head())
    print(data.info())

    # Step 2: Perform feature engineering
    print("Adding features to the data...")
    data = add_features(data, ticker)
    print("Features added successfully!")
    print(data.head())
    print(data.info())

    # Step 3: Display the final data
    print(data.head())

    # Plot the data
    plot_stock_data(data, ticker)

if __name__ == "__main__":
    main()
