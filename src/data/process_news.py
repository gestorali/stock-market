import pandas as pd
from src.data.utils import detect_language, is_mostly_non_latin, translate_text, analyze_sentiment

def process_and_save_translated_news(
    raw_file="data/raw/news_original_language.csv",
    clean_file="data/processed/news_translated_cleaned.csv",
    lang_blacklist={"zh-cn", "zh-tw", "ja", "ko", "ar", "ru"}
):
    try:
        df = pd.read_csv(raw_file)
    except FileNotFoundError:
        print(f"❌ File not found: {raw_file}")
        return

    # 🧠 Create one full text column for translation/sentiment
    df['original_text'] = (
        df['title'].astype(str) + " " +
        df['description'].astype(str) + " " +
        df['content'].astype(str)
    ).str.strip()

    # 🧹 Filter out non-supported languages
    df['detected_lang'] = df['original_text'].apply(detect_language)
    mask_lang = df['detected_lang'].isin(lang_blacklist)
    mask_nonlatin = df['original_text'].apply(is_mostly_non_latin)
    df_clean = df[~(mask_lang | mask_nonlatin)].copy()

    # 🌍 Translate and analyze sentiment
    df_clean['translated_text'] = df_clean['original_text'].apply(translate_text)
    df_clean['sentiment'] = df_clean['translated_text'].apply(analyze_sentiment)

    # 🧾 Preserve useful metadata (if exists)
    meta_columns = ['publishedAt', 'title', 'translated_text', 'sentiment', 'url', 'source.name']
    optional_meta = ['ticker', 'type', 'query', 'fetch_date']
    preserved_cols = [col for col in optional_meta if col in df_clean.columns]
    final_cols = meta_columns + preserved_cols

    # ✍️ Save cleaned/transformed file
    df_clean[final_cols].to_csv(clean_file, index=False)
    print(f"✅ Cleaned data saved to '{clean_file}'")
