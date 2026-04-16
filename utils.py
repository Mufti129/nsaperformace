def generate_recommendation(corr, importance, features):
    rec = []

    # Marketing insight
    if corr.loc['sales_quantity', 'tiktok_ad_spend'] > corr.loc['sales_quantity', 'fb_ad_spend']:
        rec.append("Increase budget allocation to TikTok Ads.")
    else:
        rec.append("Maintain or increase Facebook Ads budget.")

    # Feature importance insight
    important_features = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)

    rec.append(f"Top drivers of sales: {important_features[:3]}")

    return rec
