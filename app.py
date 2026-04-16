import streamlit as st
import pandas as pd

from analysis import load_data, clean_data, summary_stats, marketing_analysis, pricing_analysis
from model import train_model
from utils import generate_recommendation

st.title("📊 E-commerce Omnichannel Analysis Dashboard")

# Load data
df = load_data("data/ecommerce-omnichannel-analysis.csv")
df = clean_data(df)

# ===== SUMMARY =====
st.header("1. Data Summary")
st.write(summary_stats(df))
st.write("DEBUG summary:")
st.write(type(summary))
st.write(summary)
# ===== MARKETING =====
st.header("2. Marketing Analysis")
corr, df = marketing_analysis(df)
st.write("Correlation Matrix")
st.write(corr)

# ===== PRICING =====
st.header("3. Pricing Impact")
discount_impact, price_impact = pricing_analysis(df)
st.write("Discount Impact")
st.line_chart(discount_impact)

st.write("Price Impact")
st.line_chart(price_impact)

# ===== MODEL =====
st.header("4. Predictive Model")
model, rmse, r2, features, importance = train_model(df)

st.write(f"RMSE: {rmse}")
st.write(f"R2 Score: {r2}")

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

st.bar_chart(importance_df.set_index("feature"))

# ===== RECOMMENDATION =====
st.header("5. Business Recommendation")
recommendation = generate_recommendation(corr, importance, features)

for r in recommendation:
    st.write("-", r)
