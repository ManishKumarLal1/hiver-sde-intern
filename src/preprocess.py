import re
import pandas as pd

def extract_pairs(df, brand_id):
    """
    Extract one-turn pairs (customer message + brand reply) for a given brand.
    """
    df = df.copy()
    df['tweet_id'] = df['tweet_id'].astype(str).str.strip()
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].astype(str).str.strip()

    brand_replies = df[(df['inbound'] == False) & (df['author_id'] == brand_id)].copy()
    brand_replies = brand_replies[brand_replies['in_response_to_tweet_id'] != 'nan']
    brand_replies = brand_replies[brand_replies['in_response_to_tweet_id'] != '']

    customer_tweet_ids = brand_replies['in_response_to_tweet_id'].unique()
    customer_tweets = df[(df['inbound'] == True) & (df['tweet_id'].isin(customer_tweet_ids))].copy()

    pairs = customer_tweets.merge(
        brand_replies[['in_response_to_tweet_id', 'text', 'created_at']],
        left_on='tweet_id',
        right_on='in_response_to_tweet_id',
        how='inner',
        suffixes=('_cust', '_brand')
    )

    pairs = pairs.rename(columns={
        'text_cust': 'customer_msg',
        'text_brand': 'brand_reply',
        'created_at_cust': 'customer_created_at',
        'created_at_brand': 'brand_created_at'
    })

    pairs = pairs[['customer_msg', 'brand_reply', 'customer_created_at', 'brand_created_at']]
    return pairs


def clean_text(text):
    """Remove URLs, mentions, extra spaces, and normalize."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text