import pandas as pd
import re
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

def save_to_csv(df, filename, start_date=None, end_date=None):
    df = df.copy()
    if start_date:
        df["start_date"] = start_date
    if end_date:
        df["end_date"] = end_date

    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        df = df.sort_values(by="publishedAt")

    try:
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
        df.drop_duplicates(subset=["title", "publishedAt"], inplace=True)
    except FileNotFoundError:
        pass

    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} rows to '{filename}'")