import streamlit as st
import pandas as pd
import os

from analysis import load_data, clean_data, summary_stats, marketing_analysis, pricing_analysis
from model import train_model
from utils import generate_recommendation

st.title("📊 E-commerce Omnichannel Analysis Dashboard")

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

# ===== 3. PRICING =====
st.header("3. Pricing Impact")

discount_impact, price_impact = pricing_analysis(df)

if discount_impact is not None:
    st.line_chart(discount_impact)
    st.line_chart(price_impact)
else:
    st.error("❌ Pricing analysis gagal")

# ===== 4. MODEL =====
st.header("4. Predictive Model")

model, rmse, r2, features, importance = train_model(df)

st.write(f"RMSE: {rmse}")
st.write(f"R2 Score: {r2}")

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

st.bar_chart(importance_df.set_index("feature"))

# ===== 5. RECOMMENDATION =====
st.header("5. Business Recommendation")

recommendation = generate_recommendation(corr, importance, features)

for r in recommendation:
    st.write("-", r)
