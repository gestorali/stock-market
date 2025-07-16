import pandas as pd

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

    # Merge using left join to retain all price records
    df = pd.merge(df_prices, df_news, how='left', on=['ticker', 'date'])

    # Fill missing sentiment with 0 (neutral) or optionally keep NaN
    df['sentiment'] = df['sentiment'].fillna(0)

    df.sort_values(by=['ticker', 'date'], inplace=True)
    df['next_close'] = df.groupby('ticker')['Close'].shift(-1)
    df['target'] = (df['next_close'] > df['Close']).astype(int)

    df.to_csv(output_file, index=False)
    print(f"✅ Features saved to '{output_file}' with {len(df)} rows.")
    return df
