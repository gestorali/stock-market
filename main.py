from src.data.fetch_data import fetch_stock_data
from src.data.plot_data import plot_stock_data
from src.data.feature_engineering import add_features
from src.models.train_model import train_model

def main():
    # Define parameters
    ticker = "AAPL"
    start_date = "2018-01-01"
    end_date = "2023-01-01"

    # Step 1: Fetch and Process Data
    print("Fetching stock data...")
    data = fetch_stock_data(ticker, start_date, end_date)
    print("Data fetched successfully!")

    print("Adding features to the data...")
    data = add_features(data, ticker)
    print("Features added successfully!")

    # Step 2: Train the Model
    print("Training ML model...")
    model, scaler = train_model(ticker, start_date, end_date)
    print("Model training complete!")

    # Step 3: Plot Stock Data
    plot_stock_data(data, ticker)


if __name__ == "__main__":
    main()
