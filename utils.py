import pandas as pd


# ================================
# 1. BUSINESS RECOMMENDATION
# ================================
def generate_recommendation(corr, importance_df):
    rec = []

    if corr is None or importance_df is None:
        return ["Data tidak cukup untuk memberikan rekomendasi."]

    # Marketing insight
    if corr.loc['sales_quantity', 'tiktok_ad_spend'] > corr.loc['sales_quantity', 'fb_ad_spend']:
        rec.append("Alihkan sebagian anggaran ke TikTok Ads karena memiliki dampak yang lebih besar terhadap penjualan.")
    else:
        rec.append("Facebook Ads tetap menjadi channel yang kuat, pertahankan atau optimalkan investasinya.")

    # Feature importance
    top_features = importance_df.head(3)

    for _, row in top_features.iterrows():
        if row['feature'] == 'discount_pct':
            rec.append("Strategi diskon memiliki pengaruh signifikan terhadap penjualan. Gunakan promosi secara lebih terarah.")
        elif row['feature'] == 'current_sell_price':
            rec.append("Harga jual memiliki peran penting dalam penjualan. Optimalkan strategi penetapan harga.")
        elif row['feature'] == 'tiktok_ad_spend':
            rec.append("TikTok Ads merupakan pendorong utama penjualan. Pertimbangkan untuk meningkatkan skala channel ini.")
        elif row['feature'] == 'fb_ad_spend':
            rec.append("Facebook Ads berkontribusi terhadap penjualan, namun perlu dioptimalkan agar lebih efisien.")

    return rec


# ================================
# 2. DATA SUMMARY INSIGHT
# ================================
def insight_summary(df):
    if df is None or df.empty:
        return "Data tidak tersedia."

    return (
        f"Dataset terdiri dari {df.shape[0]} hari transaksi. "
        f"Rata-rata penjualan {df['sales_quantity'].mean():.2f} unit, "
        f"dengan maksimum {df['sales_quantity'].max()} dan minimum {df['sales_quantity'].min()}."
    )


# ================================
# 3. MARKETING INSIGHT
# ================================
def insight_marketing(corr):
    if corr is None:
        return "Data marketing tidak tersedia."

    fb_corr = corr.loc['sales_quantity', 'fb_ad_spend']
    tt_corr = corr.loc['sales_quantity', 'tiktok_ad_spend']

    if tt_corr > fb_corr:
        return (
            f"TikTok Ads lebih efektif (korelasi {tt_corr:.2f}) dibanding Facebook Ads ({fb_corr:.2f})."
        )
    else:
        return (
            f"Facebook Ads lebih efektif (korelasi {fb_corr:.2f}) dibanding TikTok Ads ({tt_corr:.2f})."
        )


# ================================
# 4. PRICING INSIGHT
# ================================
def insight_pricing(discount_impact, price_impact):
    if discount_impact is None or price_impact is None:
        return "Data pricing tidak tersedia."

    best_discount = discount_impact.idxmax()
    best_price = price_impact.idxmax()

    return (
        f"Diskon terbaik: {best_discount}, "
        f"Harga terbaik: {best_price}."
    )


# ================================
# 5. MODEL BASIC INSIGHT
# ================================
def insight_model(results_df, best_model_name):
    if results_df is None or results_df.empty:
        return "Model belum tersedia."

    best_rmse = results_df.iloc[0]['RMSE']
    best_r2 = results_df.iloc[0]['R2']

    return (
        f"Model terbaik adalah {best_model_name} dengan RMSE {best_rmse:.2f} "
        f"dan R² {best_r2:.2f}."
    )


# ================================
# 6. MODEL ADVANCED INSIGHT
# ================================
def insight_model_advanced(results_df, best_model_name, importance_df, df):
    if results_df is None or importance_df is None or df is None:
        return "Model belum tersedia."

    best_rmse = results_df.iloc[0]['RMSE']
    best_r2 = results_df.iloc[0]['R2']
    avg_sales = df['sales_quantity'].mean()

    rmse_pct = (best_rmse / avg_sales) * 100
    top_features = importance_df.head(3)

    insight = []

    # Model performance
    insight.append(
        f"Model terbaik adalah {best_model_name} dengan R² {best_r2:.2f} "
        f"({'lemah' if best_r2 < 0.3 else 'cukup kuat'})."
    )

    insight.append(
        f"RMSE {best_rmse:.2f} (~{rmse_pct:.1f}% dari rata-rata sales)."
    )

    # Feature importance insight
    for _, row in top_features.iterrows():
        if row['feature'] == 'base_price':
            insight.append("Harga dasar adalah faktor paling dominan.")
        elif row['feature'] == 'tiktok_ad_spend':
            insight.append("TikTok Ads adalah channel paling efektif.")
        elif row['feature'] == 'affiliate_commission_rate':
            insight.append("Program affiliate berkontribusi signifikan.")
        elif row['feature'] == 'fb_ad_spend':
            insight.append("Facebook Ads berpengaruh namun tidak dominan.")

    return " ".join(insight)
