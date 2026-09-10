import pandas as pd

def load_raw_tweets(path, nrows=None):
    """Load tweets CSV."""
    df = pd.read_csv(path, nrows=nrows)
    return df

def list_brands(df, min_tweets=1000):
    """Return brand counts."""
    print(df.columns)
    brand_counts = df[df['inbound'] == False]['author_id'].value_counts()
    return brand_counts[brand_counts >= min_tweets]