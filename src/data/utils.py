# src/data/utils.py

import pandas as pd
import re
from langdetect import detect, LangDetectException
from googletrans import Translator  # ✅ używamy googletrans-py
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
    """Mapuje kody językowe na takie, które rozumie Translator."""
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

    non_latin_chars = re.findall(r"[^\x00-\x7F]", text)

    # Jeśli zawiera chińskie/japońskie/koreańskie, zostawiamy
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return False

    return len(non_latin_chars) / max(len(text), 1) > threshold

def translate_text(text, target_lang="en", chunk_size=4000):
    """
    Tłumaczy tekst na target_lang, normalizując kod języka
    i dzieląc długie teksty na kawałki (dla googletrans-py).
    """
    if not isinstance(text, str) or not text.strip():
        return text

    try:
        detected_lang = detect_language(text)
        detected_lang = normalize_language_code(detected_lang)

        if detected_lang.lower() == target_lang.lower():
            return text  # już w odpowiednim języku

        translator = Translator()

        # Podział na kawałki (max ~5000 znaków dla API Google)
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        translated_chunks = []
        for chunk in chunks:
            try:
                result = translator.translate(chunk, src=detected_lang, dest=target_lang)
                translated_chunks.append(result.text)
            except Exception as e:
                print(f"⚠️ Translation chunk error ({detected_lang}): {e}")
                translated_chunks.append(chunk)  # fallback – zostawiamy oryginał

        return " ".join(translated_chunks)

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
