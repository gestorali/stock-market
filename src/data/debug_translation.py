import pandas as pd
from langdetect import detect, LangDetectException
import re
import os

# Set working directory to project root
os.chdir("/home/michal/PycharmProjects/stock-market")

def is_mostly_non_latin(text, threshold=0.7):
    """Check if the text is mostly non-Latin characters."""
    if not text:
        return True
    latin_chars = re.findall(r'[A-Za-z]', text)
    return len(latin_chars) / max(len(text), 1) < (1 - threshold)

def debug_language_filtering(row):
    text = row['original_text']
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'undetected'
    non_latin = is_mostly_non_latin(text)
    return pd.Series({'detected_lang': lang, 'mostly_non_latin': non_latin})

def main():
    df = pd.read_csv("news_translated.csv")
    df.fillna("", inplace=True)

    df[['detected_lang', 'mostly_non_latin']] = df.apply(debug_language_filtering, axis=1)

    unexpected = df[(df['mostly_non_latin'] == True) & (~df['detected_lang'].isin({'en', 'fr', 'de', 'nl', 'it', 'no', 'pt', 'es', 'sv'}))]
    print(f"⚠️ Unexpected articles not filtered properly: {len(unexpected)}")
    print(unexpected[['original_text', 'detected_lang']].head(3))

    # Optional: save cleaned version
    df_cleaned = df[
        (df['detected_lang'].isin({'en', 'fr', 'de', 'nl', 'it', 'no', 'pt', 'es', 'sv'})) &
        (df['mostly_non_latin'] == False)
    ]
    df_cleaned.to_csv("news_translated_cleaned.csv", index=False)
    print("✅ Cleaned file saved as 'news_translated_cleaned.csv'")

if __name__ == "__main__":
    main()
