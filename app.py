import streamlit as st
import pandas as pd
import os
from utils import generate_recommendation,insight_summary,insight_marketing,insight_pricing,insight_model
from analysis import load_data, clean_data, summary_stats, marketing_analysis, pricing_analysis
#from model import train_model
from model import train_and_compare_models
from utils import generate_recommendation
from utils import insight_model_advanced
from model import train_gradient_boosting, predict_sales

st.title("E-commerce Omnichannel Analysis Dashboard")

# ===== LOAD DATA =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "ecommerce-omnichannel-analysis.csv")

df = load_data(file_path)

if df is None:
    st.error("❌ Data gagal dimuat. Cek path atau file CSV.")
    st.stop()

df = clean_data(df)

# ===== DEBUG =====
#st.subheader("DEBUG INFO")
#st.write("Shape:", df.shape)
#st.write("Columns:", df.columns)
with st.expander("🔍 Debug Info (Klik untuk buka/tutup)"):
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns)
# ===== 1. SUMMARY =====
st.header("1. Data Summary")

summary = summary_stats(df)

if summary is not None:
    st.dataframe(summary)

    st.info(insight_summary(df))
else:
    st.error("Summary gagal")
    ###ganti#
# ===== EXTRA VALIDATION =====
#st.subheader("Missing Values")
#st.write(df.isnull().sum())
with st.expander("Missing Values (Klik untuk lihat)"):
    st.write(df.isnull().sum())
# ===== 2. MARKETING =====
st.header("2. Marketing Analysis")

corr, df = marketing_analysis(df)

if corr is not None:
    st.dataframe(corr)

    st.info(insight_marketing(corr))
else:
    st.error("Marketing gagal")
##

# ===== 3. PRICING =====
st.header("3. Pricing Impact")

discount_impact, price_impact = pricing_analysis(df)

if discount_impact is not None:

    # ===== DISCOUNT =====
    st.subheader("Discount vs Sales")

    st.write("Comparison (Bar Chart)")
    st.bar_chart(discount_impact)

    st.write("Trend View (Line Chart)")
    st.line_chart(discount_impact)

    st.dataframe(discount_impact)

    # ===== PRICE =====
    st.subheader("Price vs Sales")

    st.write("Comparison (Bar Chart)")
    st.bar_chart(price_impact)

    st.write("Trend View (Line Chart)")
    st.line_chart(price_impact)

    st.dataframe(price_impact)

else:
    st.error("Pricing analysis gagal")
st.info(insight_pricing(discount_impact, price_impact))
# ===== 4. MODEL =====
st.header("4. Predictive Model Comparison")

best_model, best_model_name, results_df, importance_df = train_and_compare_models(df)

st.subheader("Model Comparison")
st.dataframe(results_df)

st.subheader(f"Best Model: {best_model_name}")

st.subheader("Feature Importance")
st.bar_chart(importance_df.set_index("feature"))
st.info(insight_model(results_df, best_model_name))
st.subheader("Insight Model (Auto Explanation)")
st.info(insight_model_advanced(results_df, best_model_name, importance_df, df))


# ===== 5. Simulator based on best model =====
st.header("5. Sales Prediction Simulator (Gradient Boosting)")

# ===== TRAIN MODEL KHUSUS =====
gb_model, rmse, r2, features = train_gradient_boosting(df)

st.write(f"Model RMSE: {rmse:.2f}")
st.write(f"Model R2: {r2:.2f}")

st.divider()

st.subheader("Input Scenario")

# ===== INPUT =====
base_price = st.number_input("Base Price", value=50)
discount_pct = st.slider("Discount (%)", 0.0, 0.5, 0.1)

current_sell_price = base_price * (1 - discount_pct)

fb_ad_spend = st.number_input("Facebook Ads Spend", value=100)
tiktok_ad_spend = st.number_input("TikTok Ads Spend", value=100)
affiliate_commission_rate = st.slider("Affiliate Commission", 0.0, 0.3, 0.1)

st.write(f"Final Selling Price: {current_sell_price:.2f}")

# ===== PREDICT =====
if st.button("Predict Sales"):

    input_data = {
        'base_price': base_price,
        'discount_pct': discount_pct,
        'current_sell_price': current_sell_price,
        'fb_ad_spend': fb_ad_spend,
        'tiktok_ad_spend': tiktok_ad_spend,
        'affiliate_commission_rate': affiliate_commission_rate
    }

    prediction = predict_sales(gb_model, input_data)

    st.success(f"Predicted Daily Sales: {prediction:.0f} units")

    # ===== INSIGHT =====
    avg_sales = df['sales_quantity'].mean()

    if prediction > avg_sales:
        st.success("Di atas rata-rata — strategi ini berpotensi meningkatkan penjualan")
    else:
        st.warning("Di bawah rata-rata — perlu optimasi strategi")


# ===== 5. RECOMMENDATION =====
st.header("6. Business Recommendation")

#recommendation = generate_recommendation(corr, importance, features)
recommendation = generate_recommendation(corr, importance_df)
for r in recommendation:
    st.write("-", r)

