import os
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from dotenv import load_dotenv
from src.data.utils import save_to_csv

def fetch_and_save_news(ticker, start_date, end_date, raw_file="data/raw/news_original_language.csv"):
    load_dotenv()
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        print("❌ GNEWS_API_KEY not found.")
        return

    all_articles = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current_start < end:
        current_end = min(current_start + timedelta(days=10), end)
        url = (
            f"https://gnews.io/api/v4/search?q={ticker}&"
            f"from={current_start.strftime('%Y-%m-%dT%H:%M:%SZ')}&"
            f"to={current_end.strftime('%Y-%m-%dT%H:%M:%SZ')}&"
            f"token={api_key}"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            all_articles.extend(articles)
            print(f"📥 {len(articles)} articles: {current_start.date()} to {current_end.date()}")
        except Exception as e:
            print(f"❌ Fetch error: {e}")
        current_start = current_end + timedelta(days=1)
        time.sleep(1)

    if all_articles:
        df = pd.DataFrame(all_articles)
        df["ticker"] = ticker
        df["fetch_date"] = pd.Timestamp.now().date()
        save_to_csv(df, raw_file, start_date, end_date)

