# src/features/build_features.py
import pandas as pd
from src.features.technical_indicators import add_technical_indicators

def combine_news_and_prices(
    news_file="data/processed/news_translated_cleaned.csv",
    price_file="data/prices/stock_prices.csv",
    output_file="data/features/combined.csv"
):
    df_news = pd.read_csv(news_file)
    df_prices = pd.read_csv(price_file)

    df_news['publishedAt'] = pd.to_datetime(df_news['publishedAt'], errors='coerce')
    df_prices['Date'] = pd.to_datetime(df_prices['Date'], errors='coerce')

    df_news['date'] = df_news['publishedAt'].dt.date
    df_prices['date'] = df_prices['Date'].dt.date

    # Aggregate sentiment per ticker/date
    df_agg_news = df_news.groupby(['ticker', 'date']).agg({
        'sentiment': 'mean',
        'translated_text': 'count'
    }).rename(columns={'translated_text': 'news_count'}).reset_index()

    # Split general vs company sentiment
    df_general = df_agg_news[df_agg_news['ticker'] == "GENERAL"][['date', 'sentiment']].rename(columns={'sentiment': 'general_sentiment'})
    df_company = df_agg_news[df_agg_news['ticker'] != "GENERAL"]

    # Merge company sentiment to prices
    df = pd.merge(df_prices, df_company, how='left', on=['ticker', 'date'])

    # Merge general sentiment to all prices on same date
    df = pd.merge(df, df_general, how='left', on='date')

    # Fill NaNs
    df['sentiment'] = df['sentiment'].fillna(0)
    df['news_count'] = df['news_count'].fillna(0)
    df['general_sentiment'] = df['general_sentiment'].fillna(0)

    # Add technical indicators
    df = add_technical_indicators(df)

    # Target generation
    df.sort_values(by=['ticker', 'date'], inplace=True)
    df['next_close'] = df.groupby('ticker')['Close'].shift(-1)
    df['target'] = (df['next_close'] > df['Close']).astype(int)

    df.to_csv(output_file, index=False)
    print(f"✅ Features saved to '{output_file}' with {len(df)} rows.")
    return df
