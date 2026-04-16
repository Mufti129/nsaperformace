def generate_recommendation(corr, importance_df):
    rec = []

    # Marketing insight
    if corr.loc['sales_quantity', 'tiktok_ad_spend'] > corr.loc['sales_quantity', 'fb_ad_spend']:
        rec.append("Shift more budget to TikTok Ads due to higher impact on sales.")
    else:
        rec.append("Facebook Ads remains a strong channel, maintain investment.")

    # Feature importance
    top_features = importance_df.head(3)

    for _, row in top_features.iterrows():
        if row['feature'] == 'discount_pct':
            rec.append("Discount strategy significantly impacts sales. Apply targeted promotions.")
        elif row['feature'] == 'current_sell_price':
            rec.append("Pricing plays a key role. Optimize price positioning.")
        elif row['feature'] == 'tiktok_ad_spend':
            rec.append("TikTok Ads is a major driver. Consider scaling this channel.")
        elif row['feature'] == 'fb_ad_spend':
            rec.append("Facebook Ads contributes to sales. Optimize campaign efficiency.")

    return rec
