import pandas as pd


def load_data(path):
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print("ERROR LOAD DATA:", e)
        return None


def clean_data(df):
    if df is None:
        return None

    # Fill missing
    df['fb_ad_spend'] = df['fb_ad_spend'].fillna(0)
    df['tiktok_ad_spend'] = df['tiktok_ad_spend'].fillna(0)

    # Remove invalid
    df = df[df['sales_quantity'] >= 0]

    return df


def summary_stats(df):
    if df is None or df.empty:
        return None

    return df.describe()


def marketing_analysis(df):
    if df is None or df.empty:
        return None, None

    corr = df[['sales_quantity', 'fb_ad_spend', 'tiktok_ad_spend', 'affiliate_commission_rate']].corr()

    df['total_ad_spend'] = df['fb_ad_spend'] + df['tiktok_ad_spend']
    df['cost_per_unit'] = df['total_ad_spend'] / df['sales_quantity']

    return corr, df


def pricing_analysis(df):
    if df is None or df.empty:
        return None, None

    discount_impact = df.groupby('discount_pct')['sales_quantity'].mean()
    price_impact = df.groupby('current_sell_price')['sales_quantity'].mean()

    return discount_impact, price_impact
