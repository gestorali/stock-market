import pandas as pd
from langdetect import detect
import re
import os

# Set working directory to project root
os.chdir("/home/michal/PycharmProjects/stock-market")

def load_news_data(file_path="news_translated.csv"):
    """Load news data CSV with original and translated texts."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} rows from '{file_path}'")
        return df
    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found.")
        return None


def detect_language(text):
    """Detect the language of a single text string."""
    try:
        return detect(text.strip()) if isinstance(text, str) and text.strip() else "unknown"
    except LangDetectException:
        return "undetected"


def is_mostly_non_latin(text, threshold=0.3):
    """Return True if the text contains mostly non-Latin characters."""
    if not isinstance(text, str):
        return False
    non_latin_chars = re.findall(r"[^\x00-\x7F]", text)
    return len(non_latin_chars) / max(len(text), 1) > threshold


def debug_filter_articles(df):
    """Detect language, apply filters, report removed rows, and return cleaned DataFrame."""
    lang_blacklist = {"zh-cn", "zh-tw", "ja", "ko", "ar", "ru"}

    # Step 1: Detect language
    df["detected_lang"] = df["original_text"].apply(detect_language)

    # Step 2: Filter conditions
    mask_lang = df["detected_lang"].isin(lang_blacklist)
    mask_nonlatin = df["original_text"].apply(is_mostly_non_latin)

    count_lang = mask_lang.sum()
    count_nonlatin = mask_nonlatin.sum()

    # Step 3: Keep only clean rows
    df_cleaned = df[~(mask_lang | mask_nonlatin)]

    # Debug logs
    print(f"⚠️ Articles removed due to language blacklist: {count_lang}")
    print(f"⚠️ Articles removed due to mostly non-Latin characters: {count_nonlatin}")
    print(f"✅ Articles kept after filtering: {len(df_cleaned)} / {len(df)}")

    return df_cleaned


def save_cleaned_data(df_cleaned, output_path="news_translated_cleaned.csv"):
    df_cleaned.to_csv(output_path, index=False)
    print(f"✅ Cleaned file saved as '{output_path}'")


def main():
    df = load_news_data("news_translated.csv")
    if df is not None:
        df_cleaned = debug_filter_articles(df)
        save_cleaned_data(df_cleaned)


if __name__ == "__main__":
    main()
