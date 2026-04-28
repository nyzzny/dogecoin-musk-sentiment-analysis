"""
Full Data Preparation Pipeline for Dogecoin Sentiment Analysis (DAILY VERSION) - FIXED
- Musk tweets: daily counts + sentiment (enhanced VADER)
- Reddit: daily post count + sentiment (enhanced VADER)
Output: Master_Analysis_Dataset_Daily_v2.csv

FIXES:
- Reddit avg sentiment is now NaN on days with zero posts (instead of incorrectly filled 0).
- Merge function preserves NaNs to avoid misrepresenting "no discussion" as "neutral sentiment".
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================================
# 1. Configuration
# ============================================================================
START_DATE = '2017-01-01'
END_DATE = '2024-04-13'
TICKER = 'DOGE-USD'
MUSK_RAW_FILE = 'all_musk_posts.csv'
REDDIT_RAW_FILE = 'r_dogecoin_posts.jsonl'
FINAL_OUTPUT_FILE = 'Master_Analysis_Dataset_Daily_v2_FIXED.csv'   # 输出文件名加上 FIXED 以便区分

REDDIT_CHUNK_SIZE = 100000

DOGE_KEYWORDS = ['doge', 'dogecoin', 'shiba', 'moon', 'crypto', 'coin']
HIGH_IMPACT_LIKE_THRESHOLD = 100000

# 加密货币情感词典（增强版）
CRYPTO_LEXICON = {
    'moon': 3.0, 'wen': 2.0, 'hodl': 2.5, 'pump': 2.5, '🚀': 2.0,
    '🐕': 1.5, '💎🙌': 2.5, 'doge': 2.0, 'dogecoin': 2.0, 'shiba': 1.5,
    'to the moon': 3.5, 'bullish': 2.0, 'buy': 1.0,
    'dump': -2.5, 'rekt': -3.0, 'bearish': -2.0, 'sell': -1.0,
    'crash': -2.5, 'scam': -3.0,
}

# ============================================================================
# 2. Dogecoin Price Data (Daily)
# ============================================================================
def clean_yfinance_columns(df):
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_cols.append(col[0].capitalize())
        else:
            new_cols.append(col.capitalize())
    df.columns = new_cols
    return df

def build_daily_price_data():
    print("\n[1] Downloading daily price data...")
    daily = yf.download(TICKER, start=START_DATE, end=END_DATE, interval='1d')
    if daily.empty:
        raise RuntimeError("Daily price download failed.")
    daily = clean_yfinance_columns(daily)
    daily.index = daily.index.tz_localize('UTC')
    daily = daily[['Close']].copy()
    daily['daily_return'] = daily['Close'].pct_change()
    daily = daily.dropna()
    print(f"   -> Daily price data ready: {len(daily)} records.")
    return daily

# ============================================================================
# 3. Elon Musk Tweets Cleaning (Daily aggregation with sentiment)
# ============================================================================
def clean_musk_tweets_daily():
    print("\n[2] Cleaning Musk tweets (daily aggregation with sentiment)...")
    df = pd.read_csv(MUSK_RAW_FILE, low_memory=False)

    date_col = 'createdAt'
    text_col = 'fullText'
    like_col = 'likeCount'
    retweet_col = 'retweetCount'
    is_retweet_col = 'isRetweet'

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)

    df = df[df[is_retweet_col] != True]
    df = df[df[is_retweet_col] != 'True']

    df[date_col] = df[date_col].dt.tz_convert('UTC')
    df = df[(df[date_col] >= START_DATE) & (df[date_col] <= END_DATE)]

    df = df[[date_col, text_col, like_col, retweet_col]].copy()
    df.rename(columns={
        date_col: 'datetime',
        text_col: 'text',
        like_col: 'likes',
        retweet_col: 'retweets'
    }, inplace=True)

    df.fillna(0, inplace=True)
    df[['likes', 'retweets']] = df[['likes', 'retweets']].astype(int)

    df['is_doge_related'] = df['text'].str.contains('|'.join(DOGE_KEYWORDS), case=False, na=False)
    df['is_high_impact'] = (df['likes'] > HIGH_IMPACT_LIKE_THRESHOLD) & (df['is_doge_related'])

    # 情感分析
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(CRYPTO_LEXICON)
    df['sentiment'] = df['text'].apply(lambda txt: analyzer.polarity_scores(txt)['compound'])

    # 日度聚合
    df.set_index('datetime', inplace=True)
    daily_musk = df.resample('D').agg(
        musk_high_impact_tweets=('is_high_impact', 'sum'),
        musk_avg_sentiment=('sentiment', 'mean'),
        musk_sentiment_std=('sentiment', 'std'),
        musk_positive_count=('sentiment', lambda x: (x > 0.05).sum()),
        musk_negative_count=('sentiment', lambda x: (x < -0.05).sum()),
        musk_tweet_count=('text', 'count')
    )
    # 填充 NaN (当某天没有推文时，计数为0，情感均值填0是合理的，因为无推文 = 无情绪影响)
    daily_musk['musk_avg_sentiment'] = daily_musk['musk_avg_sentiment'].fillna(0)
    daily_musk['musk_sentiment_std'] = daily_musk['musk_sentiment_std'].fillna(0)
    daily_musk['musk_positive_count'] = daily_musk['musk_positive_count'].fillna(0).astype(int)
    daily_musk['musk_negative_count'] = daily_musk['musk_negative_count'].fillna(0).astype(int)
    daily_musk['musk_tweet_count'] = daily_musk['musk_tweet_count'].fillna(0).astype(int)

    print(f"   -> Musk data: {len(df)} tweets, {len(daily_musk)} days with non‑zero tweets.")
    return daily_musk

# ============================================================================
# 4. Reddit Data Processing with Enhanced VADER (Daily aggregation) [FIXED]
# ============================================================================
def process_reddit_daily():
    print("\n[3] Processing Reddit data with enhanced VADER (daily aggregation)...")
    
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(CRYPTO_LEXICON)
    
    start_ts = pd.Timestamp(START_DATE, tz='UTC').timestamp()
    end_ts = pd.Timestamp(END_DATE, tz='UTC').timestamp()

    chunk_iter = pd.read_json(REDDIT_RAW_FILE, lines=True, chunksize=REDDIT_CHUNK_SIZE,
                              dtype={'title': 'string', 'selftext': 'string'})

    all_daily_aggs = []
    total_rows = 0
    start_time = time.time()

    for i, chunk in enumerate(chunk_iter, 1):
        print(f"   Chunk {i}...", end=' ')
        chunk = chunk[pd.to_numeric(chunk['created_utc'], errors='coerce').notna()]
        chunk['created_utc'] = chunk['created_utc'].astype(float)
        chunk = chunk[(chunk['created_utc'] >= start_ts) & (chunk['created_utc'] <= end_ts)]
        if chunk.empty:
            print("skipped (no rows in date range).")
            continue

        chunk['datetime'] = pd.to_datetime(chunk['created_utc'], unit='s', utc=True)
        chunk['full_text'] = chunk['title'].fillna('') + ' ' + chunk['selftext'].fillna('')

        # 情感分析
        mask = chunk['full_text'].str.strip() != ''
        chunk.loc[mask, 'sentiment'] = chunk.loc[mask, 'full_text'].apply(
            lambda txt: analyzer.polarity_scores(txt)['compound'])
        chunk['sentiment'] = chunk['sentiment'].fillna(0)

        # 日度聚合
        chunk.set_index('datetime', inplace=True)
        daily = chunk.resample('D').agg(
            reddit_post_count=('full_text', 'count'),
            reddit_sentiment_sum=('sentiment', 'sum')
        )
        daily = daily[daily['reddit_post_count'] > 0]
        all_daily_aggs.append(daily)
        total_rows += len(chunk)
        print(f"kept {len(daily)} active days.")

    if not all_daily_aggs:
        print("   No Reddit data found. Creating empty series.")
        full_idx = pd.date_range(start=START_DATE, end=END_DATE, freq='D', tz='UTC')
        # 返回全NaN的情感均值，post count为0
        return pd.DataFrame(index=full_idx, columns=['reddit_post_count', 'reddit_avg_sentiment']).fillna({'reddit_post_count':0})

    combined = pd.concat(all_daily_aggs)
    final_daily = combined.groupby(combined.index).agg(
        reddit_post_count=('reddit_post_count', 'sum'),
        reddit_sentiment_sum=('reddit_sentiment_sum', 'sum')
    )
    final_daily['reddit_avg_sentiment'] = final_daily['reddit_sentiment_sum'] / final_daily['reddit_post_count']
    final_daily.drop(columns=['reddit_sentiment_sum'], inplace=True)

    # [FIXED] 重新索引到完整日期范围，并将 post_count 填充为 0，avg_sentiment 填充为 NaN
    full_idx = pd.date_range(start=START_DATE, end=END_DATE, freq='D', tz='UTC')
    final_daily = final_daily.reindex(full_idx)
    # reddit_post_count 用 0 填充，表示当天无帖子
    final_daily['reddit_post_count'] = final_daily['reddit_post_count'].fillna(0).astype(int)
    # reddit_avg_sentiment 保持 NaN，表示当天无情绪数据（而不是错误地填0）
    # 这样分析者可以自行决定如何处理缺失值（如前向填充、插值或丢弃）

    elapsed = (time.time() - start_time) / 60
    print(f"   Reddit processing finished. {total_rows} rows, {elapsed:.2f} minutes.")
    return final_daily

# ============================================================================
# 5. Final Merge [FIXED]
# ============================================================================
def merge_all_datasets_daily(price_df, musk_daily, reddit_daily):
    print("\n[4] Merging daily datasets...")
    master = price_df[['Close', 'daily_return']].copy()
    master = master.join(musk_daily).join(reddit_daily)

    # Musk 相关特征的缺失值处理（无推文日填0，符合逻辑）
    master['musk_high_impact_tweets'] = master['musk_high_impact_tweets'].fillna(0).astype(int)
    master['musk_avg_sentiment'] = master['musk_avg_sentiment'].fillna(0)
    master['musk_sentiment_std'] = master['musk_sentiment_std'].fillna(0)
    master['musk_positive_count'] = master['musk_positive_count'].fillna(0).astype(int)
    master['musk_negative_count'] = master['musk_negative_count'].fillna(0).astype(int)
    master['musk_tweet_count'] = master['musk_tweet_count'].fillna(0).astype(int)

    # [FIXED] Reddit post count 填0，但 avg_sentiment 保留 NaN，不自动填0
    master['reddit_post_count'] = master['reddit_post_count'].fillna(0).astype(int)
    # reddit_avg_sentiment 已经是 NaN 在无数据日，这里不做填充，让分析者决定

    return master

# ============================================================================
# 6. Main Execution
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DOGECOIN SENTIMENT ANALYSIS – DAILY DATA PREPARATION PIPELINE (FIXED VERSION)")
    print("=" * 60)

    price_data = build_daily_price_data()
    musk_data = clean_musk_tweets_daily()
    reddit_data = process_reddit_daily()
    final_df = merge_all_datasets_daily(price_data, musk_data, reddit_data)

    final_df.to_csv(FINAL_OUTPUT_FILE)
    print("\n" + "=" * 60)
    print(f"SUCCESS! Fixed daily dataset saved to: {FINAL_OUTPUT_FILE}")
    print(f"Total records: {len(final_df)}")
    print(f"Time range: {final_df.index.min()} → {final_df.index.max()}")
    print("=" * 60)
    print("\nPreview of the final dataset:")
    print(final_df.head())
    print("\nDataset info:")
    final_df.info()
    print("\n[NOTE] 'reddit_avg_sentiment' is NaN on days with zero Reddit posts.")