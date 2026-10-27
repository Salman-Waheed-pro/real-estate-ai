import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib, numpy as np
from src.recommend import recommend
from src.preprocess import load_and_clean

st.set_page_config(page_title="Pakistan Real Estate AI", layout="wide")
st.title("Pakistan Real Estate — AI Price Predictor & Investment Recommender")

df_raw = load_and_clean()

st.sidebar.header("Your Preferences")

city = st.sidebar.selectbox("City", sorted(df_raw["city"].unique()))
property_type = st.sidebar.selectbox("Property Type", sorted(df_raw["property_type"].unique()))
budget = st.sidebar.number_input("Budget (PKR)", min_value=500_000,
                                         max_value=60_000_000, value=15_000_000, step=500_000)
min_beds = st.sidebar.slider("Minimum Bedrooms", 1, 8, 2)
min_baths = st.sidebar.slider("Minimum Bathrooms", 1, 6, 1)
min_area = st.sidebar.slider("Minimum Area (Marla)", 1, 50, 4)
top_n = st.sidebar.slider("Results to Show", 3, 15, 5)

run = st.sidebar.button("Find Best Investments", type="primary")


st.subheader("Predict a Property Price")
col1, col2, col3 = st.columns(3)

with col1:
    p_city = st.selectbox("City", sorted(df_raw["city"].unique()), key="p_city")
    p_type = st.selectbox("Type", sorted(df_raw["property_type"].unique()), key="p_type")
with col2:
    p_location = st.selectbox("Location", sorted(df_raw[df_raw["city"] == p_city]["location"].unique()))
    p_beds = st.number_input("Bedrooms",  1, 10, 3)
with col3:
    p_baths = st.number_input("Bathrooms", 1, 10, 2)
    p_area = st.number_input("Area (Marla)", 1.0, 500.0, 5.0)

if st.button("Predict Price"):
    rf = joblib.load("models/rf_model.pkl")
    le_dict = joblib.load("models/label_encoders.pkl")
    from src.recommend import predict_price
    price = predict_price(rf, le_dict, p_type, p_location, p_city, p_baths, p_beds, p_area)
    st.success(f"Estimated Price: **PKR {price:,.0f}** ({price/1_000_000:.2f}M)")


st.divider()


if run:
    st.subheader(f"Top {top_n} Investment Opportunities in {city}")

    with st.spinner("Analysing properties..."):
        results, status = recommend(budget, city, property_type,
                                    min_beds, min_baths, min_area, top_n)

    if status != "success" or results.empty:
        st.warning("No properties found. Try relaxing your filters.")
    else:
        display = results.copy()
        display["price"] = display["price"].apply(lambda x: f"PKR {x:,.0f}")
        display["predicted_price"] = display["predicted_price"].apply(lambda x: f"PKR {x:,.0f}")
        display["investment_score"]= display["investment_score"].apply(lambda x: f"{x:.1f}/100")
        st.dataframe(display, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [f"{r['location'][:18]}..." if len(r["location"]) > 18
                   else r["location"] for _, r in results.iterrows()]
        ax.barh(labels, results["investment_score"], color="purple", alpha=0.8)
        ax.set_xlabel("Investment Score (out of 100)")
        ax.set_title("Investment Score by Location")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        # --- price vs predicted chart ---
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        x = np.arange(len(results))
        width = 0.35
        ax2.bar(x - width/2, results["price"] / 1e6, width, label="Listed Price",     color="purple", alpha=0.8)
        ax2.bar(x + width/2, results["predicted_price"] / 1e6, width, label="AI Fair Price",    color="orange", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=30, ha="right")
        ax2.set_ylabel("Price (Million PKR)")
        ax2.set_title("Listed Price vs AI Predicted Fair Price")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)