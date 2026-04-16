def generate_recommendation(corr, importance_df):
    rec = []

    # Marketing insight
    if corr.loc['sales_quantity', 'tiktok_ad_spend'] > corr.loc['sales_quantity', 'fb_ad_spend']:
        #rec.append("Shift more budget to TikTok Ads due to higher impact on sales.")
        rec.append("Alihkan sebagian anggaran ke TikTok Ads karena memiliki dampak yang lebih besar terhadap penjualan.")
    else:
        #rec.append("Facebook Ads remains a strong channel, maintain investment.")
        rec.append("Facebook Ads tetap menjadi channel yang kuat, pertahankan atau optimalkan investasinya.")
    # Feature importance
    top_features = importance_df.head(3)

    for _, row in top_features.iterrows():
        if row['feature'] == 'discount_pct':
            #rec.append("Discount strategy significantly impacts sales. Apply targeted promotions.")
            rec.append("Strategi diskon memiliki pengaruh signifikan terhadap penjualan. Gunakan promosi secara lebih terarah.")
        elif row['feature'] == 'current_sell_price':
            #rec.append("Pricing plays a key role. Optimize price positioning.")
            rec.append("Harga jual memiliki peran penting dalam penjualan. Optimalkan strategi penetapan harga.")
        elif row['feature'] == 'tiktok_ad_spend':
            #rec.append("TikTok Ads is a major driver. Consider scaling this channel.")
            rec.append("TikTok Ads merupakan pendorong utama penjualan. Pertimbangkan untuk meningkatkan skala channel ini.")
        elif row['feature'] == 'fb_ad_spend':
        elif row['feature'] == 'fb_ad_spend':
            #rec.append("Facebook Ads contributes to sales. Optimize campaign efficiency.")
            rec.append("Facebook Ads berkontribusi terhadap penjualan, namun perlu dioptimalkan agar lebih efisien.")

    return rec

def insight_summary(df):
    if df is None or df.empty:
        return "Data tidak tersedia."

    total_days = df.shape[0]
    avg_sales = df['sales_quantity'].mean()
    max_sales = df['sales_quantity'].max()
    min_sales = df['sales_quantity'].min()

    return (
        f"Dataset terdiri dari {total_days} hari transaksi. "
        f"Rata-rata penjualan harian adalah {avg_sales:.2f} unit, "
        f"dengan penjualan tertinggi {max_sales} dan terendah {min_sales}."
    )


def insight_marketing(corr):
    if corr is None:
        return "Data marketing tidak tersedia."

    fb_corr = corr.loc['sales_quantity', 'fb_ad_spend']
    tt_corr = corr.loc['sales_quantity', 'tiktok_ad_spend']

    if tt_corr > fb_corr:
        return (
            f"TikTok Ads memiliki korelasi lebih tinggi ({tt_corr:.2f}) dibanding Facebook Ads ({fb_corr:.2f}), "
            "menunjukkan bahwa TikTok lebih efektif dalam mendorong penjualan."
        )
    else:
        return (
            f"Facebook Ads memiliki korelasi lebih tinggi ({fb_corr:.2f}) dibanding TikTok Ads ({tt_corr:.2f}), "
            "menunjukkan bahwa Facebook lebih efektif dalam mendorong penjualan."
        )


def insight_pricing(discount_impact, price_impact):
    if discount_impact is None or price_impact is None:
        return "Data pricing tidak tersedia."

    best_discount = discount_impact.idxmax()
    best_price = price_impact.idxmax()

    return (
        f"Diskon {best_discount} memberikan rata-rata penjualan tertinggi. "
        f"Harga {best_price} memiliki performa penjualan terbaik dibanding harga lainnya."
    )


def insight_model(results_df, best_model_name):
    if results_df is None:
        return "Model belum tersedia."

    best_rmse = results_df.iloc[0]['RMSE']
    best_r2 = results_df.iloc[0]['R2']

def insight_model_advanced(results_df, best_model_name, importance_df, df):
    if results_df is None:
        return "Model belum tersedia."

    best_rmse = results_df.iloc[0]['RMSE']
    best_r2 = results_df.iloc[0]['R2']
    avg_sales = df['sales_quantity'].mean()

    # RMSE percentage
    rmse_pct = (best_rmse / avg_sales) * 100

    # Top features
    top_features = importance_df.head(3)

    insight = []

    # Model performance
    insight.append(
        f"Model terbaik adalah {best_model_name} dengan R² {best_r2:.2f} "
        f"yang tergolong {'lemah' if best_r2 < 0.3 else 'cukup kuat'}."
    )

    insight.append(
        f"RMSE sebesar {best_rmse:.2f} (~{rmse_pct:.1f}% dari rata-rata sales), "
        "yang masih dapat diterima untuk kebutuhan bisnis."
    )

    # Feature importance
    for _, row in top_features.iterrows():
        if row['feature'] == 'base_price':
            insight.append("Harga dasar adalah faktor paling dominan dalam menentukan penjualan.")
        elif row['feature'] == 'tiktok_ad_spend':
            insight.append("TikTok Ads merupakan channel marketing paling efektif.")
        elif row['feature'] == 'affiliate_commission_rate':
            insight.append("Program affiliate memberikan kontribusi signifikan terhadap sales.")
        elif row['feature'] == 'fb_ad_spend':
            insight.append("Facebook Ads memiliki pengaruh namun tidak dominan.")

    return " ".join(insight)

    return (
        f"Model terbaik adalah {best_model_name} dengan RMSE {best_rmse:.2f} "
        f"dan R² {best_r2:.2f}, menunjukkan performa prediksi yang cukup baik."
    )
