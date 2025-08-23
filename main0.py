# main.py
import argparse
from src.data.fetch_news import fetch_and_save_news
from src.data.fetch_prices import fetch_and_save_stock_data
from src.data.process_news import process_and_save_translated_news
from src.features.build_features import combine_news_and_prices
from src.features.technical_indicators import run_scaling_pipeline
from src.models.train_model import train_model

def main():
    parser = argparse.ArgumentParser(description="News & Stock ML Pipeline")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["fetch_news", "fetch_prices", "process_news", "combine", "train", "scale", "all"],
                        help="Which step to run")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start_date", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-07-04", help="End date (YYYY-MM-DD)")
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'logistic_regression', 'gradient_boosting', 'xgboost'],
                        help="Specify model to train (default: random_forest)")
    parser.add_argument('--general_news', action='store_true',
                        help="Fetch general financial news (not just company-specific)")

    args = parser.parse_args()

    if args.mode == "fetch_news":
        # Company-specific news
        fetch_and_save_news(args.ticker, args.start_date, args.end_date)
        if args.general_news:
            # General financial news
            fetch_and_save_news(
                ticker="GENERAL",
                start_date=args.start_date,
                end_date=args.end_date,
                query="stock market OR economy OR inflation OR interest rates",
                news_type="general"
            )

    elif args.mode == "fetch_prices":
        fetch_and_save_stock_data(args.ticker, args.start_date, args.end_date)

    elif args.mode == "process_news":
        process_and_save_translated_news()

    elif args.mode == "combine":
        combine_news_and_prices()

    elif args.mode == "scale":
        run_scaling_pipeline()

    elif args.mode == "train":
        train_model(args.model)

    elif args.mode == "all":
        fetch_and_save_news(args.ticker, args.start_date, args.end_date)
        fetch_and_save_news(
            ticker="GENERAL",
            start_date=args.start_date,
            end_date=args.end_date,
            query="stock market OR economy OR inflation OR interest rates",
            news_type="general"
        )
        fetch_and_save_stock_data(args.ticker, args.start_date, args.end_date)
        process_and_save_translated_news()
        combine_news_and_prices()
        run_scaling_pipeline()
        train_model()

if __name__ == "__main__":
    main()
