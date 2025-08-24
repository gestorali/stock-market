# src/data/utils.py

import pandas as pd
import re
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Mapowanie nietypowych kodów językowych na poprawne
LANGUAGE_CODE_MAP = {
    "zh-cn": "zh-CN",  # Chinese simplified
    "zh-tw": "zh-TW",  # Chinese traditional
    "pt-br": "pt",     # Brazilian Portuguese → ogólny portugalski
    "jp": "ja",        # czasem detektor języka zwraca 'jp' zamiast 'ja'
    "kr": "ko",        # analogicznie dla koreańskiego
    # możesz dopisać inne, jeśli wyjdą w testach
}

def normalize_language_code(lang_code: str) -> str:
    """Mapuje kody językowe na takie, które rozumie GoogleTranslator."""
    if not isinstance(lang_code, str):
        return "unknown"
    return LANGUAGE_CODE_MAP.get(lang_code.lower(), lang_code)

def detect_language(text):
    """Detekcja języka z normalizacją kodu."""
    try:
        if isinstance(text, str) and text.strip():
            detected = detect(text.strip())
            return normalize_language_code(detected)
        return "unknown"
    except LangDetectException:
        return "undetected"

def is_mostly_non_latin(text, threshold=0.3):
    """
    Returns True if text is mostly non-Latin characters,
    BUT keeps CJK (Chinese/Japanese/Korean) since we can translate them.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    # Count non-Latin
    non_latin_chars = re.findall(r"[^\x00-\x7F]", text)

    # If text is mostly Chinese/Japanese/Korean, allow it
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return False  # keep CJK text

    # Otherwise, treat as junk if non-Latin ratio is too high
    return len(non_latin_chars) / max(len(text), 1) > threshold

def translate_text(text, target_lang="en"):
    """Tłumaczy tekst na target_lang, uwzględniając normalizację kodu języka."""
    try:
        detected_lang = detect_language(text)
        if detected_lang.lower() == target_lang:
            return text
        return GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f"⚠️ Translation error ({detected_lang}): {e}")
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