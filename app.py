import streamlit as st
import pandas as pd
import os
from utils import generate_recommendation,insight_summary,insight_marketing,insight_pricing,insight_model
from analysis import load_data, clean_data, summary_stats, marketing_analysis, pricing_analysis
#from model import train_model
from model import train_and_compare_models
from utils import generate_recommendation
from utils import insight_model_advanced

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
st.subheader("DEBUG INFO")
st.write("Shape:", df.shape)
st.write("Columns:", df.columns)

# ===== 1. SUMMARY =====
st.header("1. Data Summary")

summary = summary_stats(df)

if summary is not None:
    st.dataframe(summary)
else:
    st.error("❌ Summary gagal ditampilkan (data kosong / error)")
###ganti
st.header("1. Data Summary")

summary = summary_stats(df)

if summary is not None:
    st.dataframe(summary)

    st.info(insight_summary(df))
else:
    st.error("Summary gagal")
    ###ganti#
# ===== EXTRA VALIDATION =====
st.subheader("Missing Values")
st.write(df.isnull().sum())

# ===== 2. MARKETING =====
st.header("2. Marketing Analysis")

corr, df = marketing_analysis(df)

if corr is not None:
    st.write("Correlation Matrix")
    st.dataframe(corr)
else:
    st.error("❌ Marketing analysis gagal")
##
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

    st.write("📊 Comparison (Bar Chart)")
    st.bar_chart(discount_impact)

    st.write("📈 Trend View (Line Chart)")
    st.line_chart(discount_impact)

    st.dataframe(discount_impact)

    # ===== PRICE =====
    st.subheader("Price vs Sales")

    st.write("📊 Comparison (Bar Chart)")
    st.bar_chart(price_impact)

    st.write("📈 Trend View (Line Chart)")
    st.line_chart(price_impact)

    st.dataframe(price_impact)

else:
    st.error("❌ Pricing analysis gagal")
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
# ===== 5. RECOMMENDATION =====
st.header("5. Business Recommendation")

#recommendation = generate_recommendation(corr, importance, features)
recommendation = generate_recommendation(corr, importance_df)
for r in recommendation:
    st.write("-", r)
