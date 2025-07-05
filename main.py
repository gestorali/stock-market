import os
import time
import pandas as pd
import re
from datetime import datetime, timedelta
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests

# === CONFIG ===

# mode = "fetch"  # Change to "process" to run processing phase
# mode = "fetch"       # Fetch and save original data
mode = "process"    # Load, translate, analyze
ticker = "AAPL"
start_date = "2025-06-01"
end_date = "2025-07-04"
RAW_FILE = "news_original_language.csv"
CLEAN_FILE = "news_translated_cleaned.csv"
lang_blacklist = {"zh-cn", "zh-tw", "ja", "ko", "ar", "ru"}

# === UTILS ===

def detect_language(text):
    try:
        return detect(text.strip()) if isinstance(text, str) and text.strip() else "unknown"
    except LangDetectException:
        return "undetected"

def is_mostly_non_latin(text, threshold=0.3):
    if not isinstance(text, str):
        return False
    non_latin_chars = re.findall(r"[^\x00-\x7F]", text)
    return len(non_latin_chars) / max(len(text), 1) > threshold

def translate_text(text, target_lang="en"):
    try:
        detected_lang = detect(text)
        if detected_lang.lower() == target_lang:
            return text
        return GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return text

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def save_to_csv(df_new, filename, start_date=None, end_date=None):
    df_new = df_new.copy()

    df_new["start_date"] = start_date
    df_new["end_date"] = end_date

    if "publishedAt" in df_new.columns:
        df_new["publishedAt"] = pd.to_datetime(df_new["publishedAt"], errors="coerce")

    try:
        df_existing = pd.read_csv(filename, parse_dates=["publishedAt"])
        combined_df = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"🔄 Appended {len(df_new)} rows to existing {len(df_existing)} rows")
    except FileNotFoundError:
        combined_df = df_new
        print(f"📄 File '{filename}' not found — creating new with {len(df_new)} rows")

    combined_df.drop_duplicates(subset=["title", "publishedAt"], keep="last", inplace=True)
    combined_df.sort_values(by="publishedAt", inplace=True)

    combined_df.to_csv(filename, index=False)
    print(f"✅ Saved {len(combined_df)} total rows to '{filename}', sorted by publishedAt.")

def load_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} rows from '{filepath}'")
        return df
    except FileNotFoundError:
        print(f"❌ File '{filepath}' not found.")
        return None

# === FETCH ===

def get_news_data(ticker, start_date, end_date, api_key, max_requests=100, days_per_request=10):
    all_articles = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    requests_made = 0

    while current_start < end and requests_made < max_requests:
        current_end = min(current_start + timedelta(days=days_per_request), end)
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
            print(f"📥 Fetched {len(articles)} articles: {current_start.date()} to {current_end.date()}")
            requests_made += 1
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data: {e}")
            break

        current_start = current_end + timedelta(days=1)
        time.sleep(1)

    print(f"📦 Total fetched: {len(all_articles)} articles from {requests_made} requests.")
    return all_articles

# === PROCESS ===

def process_and_filter_articles(df):
    df['original_text'] = (
        df['title'].astype(str) + " " +
        df['description'].astype(str) + " " +
        df['content'].astype(str)
    ).str.strip()

    df['detected_lang'] = df['original_text'].apply(detect_language)
    mask_lang = df['detected_lang'].isin(lang_blacklist)
    mask_nonlatin = df['original_text'].apply(is_mostly_non_latin)

    count_lang = mask_lang.sum()
    count_nonlatin = mask_nonlatin.sum()
    df_clean = df[~(mask_lang | mask_nonlatin)].copy()

    print(f"⚠️ Removed due to blacklisted languages: {count_lang}")
    print(f"⚠️ Removed due to mostly non-Latin: {count_nonlatin}")
    print(f"✅ Remaining after filtering: {len(df_clean)} / {len(df)}")

    df_clean['translated_text'] = df_clean['original_text'].apply(translate_text)
    df_clean['sentiment'] = df_clean['translated_text'].apply(analyze_sentiment)

    df_clean = df_clean.sort_values(by="publishedAt")

    df_clean.to_csv(CLEAN_FILE, index=False)
    print(f"✅ Cleaned data saved to '{CLEAN_FILE}'")

    return df_clean

# === MAIN ===

def main():
    load_dotenv()
    api_key = os.getenv("GNEWS_API_KEY")

    if not api_key and mode == "fetch":
        print("❌ GNEWS_API_KEY not found in environment.")
        return

    if mode == "fetch":
        print("🚀 Fetching news data...")
        articles = get_news_data(ticker, start_date, end_date, api_key)
        if articles:
            df = pd.DataFrame(articles)
            df["ticker"] = ticker
            df["fetch_date"] = pd.Timestamp.now().date()
            save_to_csv(df, RAW_FILE, start_date=start_date, end_date=end_date)
        else:
            print("❌ No articles fetched.")

    elif mode == "process":
        print("🔍 Processing and translating news data...")
        df = load_from_csv(RAW_FILE)
        if df is not None:
            df_clean = process_and_filter_articles(df)
            print(df_clean[['original_text', 'translated_text', 'sentiment']].head())
        else:
            print("❌ No data to process.")

    else:
        print(f"❌ Unknown mode: '{mode}' — use 'fetch' or 'process'")


if __name__ == "__main__":
    main()
