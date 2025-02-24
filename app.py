import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

@st.cache_resource
def load_model(model_filename):
    with open(model_filename, "rb") as f:
        return pickle.load(f)

ep_model = load_model("ep_model.pkl")  # Earnings Prediction Model
pp_model = load_model("pp_model.pkl")  # Potential Earnings Prediction Model

st.title("📊 Earnings vs. Potential Earnings Prediction App")

uploaded_file = st.file_uploader("Upload your hourly data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Uploaded Successfully!")

    features = [
        'bill_total_net', 'bill_total_billed', 'order_type_encoded',
        'payment_amount', 'num_people', 'holiday', 'day_of_week',
        'hour_of_day', 'is_weekend', 'payment_per_person'
    ]

    # Ensure required columns exist
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        st.error(f"❌ Missing required columns: {missing_features}")
    else:
        # Fill missing values
        df[features] = df[features].fillna(0)

        # Predict earnings using both models
        df["predicted_actual_earnings"] = ep_model.predict(df[features])
        df["predicted_potential_earnings"] = pp_model.predict(df[features])

        # Calculate difference
        df["potential_vs_actual"] = df["predicted_potential_earnings"] - df["predicted_actual_earnings"]

        # Display predictions
        st.subheader("📊 Model Predictions")
        st.write(df[["hour_of_day", "day_of_week", "predicted_actual_earnings", "predicted_potential_earnings", "potential_vs_actual"]])

        # Plot Earnings vs. Potential Earnings
        st.subheader("📈 Earnings vs. Potential Earnings Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["hour_of_day"], df["predicted_actual_earnings"], label="Actual Earnings", marker="o")
        ax.plot(df["hour_of_day"], df["predicted_potential_earnings"], label="Potential Earnings", linestyle="dashed", marker="s")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Earnings")
        ax.set_title("Predicted Actual vs. Potential Earnings")
        ax.legend()
        st.pyplot(fig)

        # Show Top 5 Underperforming Hours
        st.subheader("⚠️ Top 5 Hours with Highest Missed Revenue")
        top_underperformance = df.nlargest(5, "potential_vs_actual")[["hour_of_day", "day_of_week", "predicted_actual_earnings", "predicted_potential_earnings", "potential_vs_actual"]]
        st.write(top_underperformance)

        # Feature Importance for Both Models
        st.subheader("📌 Feature Importance Comparison")

        # Earnings Prediction Model Feature Importance
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(ep_model, importance_type="gain", ax=ax)
        ax.set_title("Feature Importance - Earnings Prediction Model")
        st.pyplot(fig)

        # Potential Earnings Model Feature Importance
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(pp_model, importance_type="gain", ax=ax)
        ax.set_title("Feature Importance - Potential Earnings Prediction Model")
        st.pyplot(fig)

        st.success("✅ Predictions Completed!")
