import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def clean_data(df):
    # Handle missing values
    df['fb_ad_spend'] = df['fb_ad_spend'].fillna(0)
    df['tiktok_ad_spend'] = df['tiktok_ad_spend'].fillna(0)
    
    # Remove negative values
    df = df[df['sales_quantity'] >= 0]
    
    return df


def summary_stats(df):
    summary = df.describe()
####POIN 2####
def marketing_analysis(df):
    # Correlation
    corr = df[['sales_quantity', 'fb_ad_spend', 'tiktok_ad_spend', 'affiliate_commission_rate']].corr()

    # Efficiency metrics
    df['total_ad_spend'] = df['fb_ad_spend'] + df['tiktok_ad_spend']
    df['cost_per_unit'] = df['total_ad_spend'] / df['sales_quantity']

    return corr, df
    return summary

####PRICING
def pricing_analysis(df):
    discount_impact = df.groupby('discount_pct')['sales_quantity'].mean()
    price_impact = df.groupby('current_sell_price')['sales_quantity'].mean()
    
    return discount_impact, price_impact
