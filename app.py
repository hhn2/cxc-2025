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

    # Ensure datetime format
    df['bill_paid_at_local'] = pd.to_datetime(df['bill_paid_at_local'])

    # Define expected features (Ensure these match the trained model)
    expected_features = [
        'bill_total_net', 'bill_total_billed',
        'payment_amount', 'num_people', 'holiday', 'day_of_week',
        'hour_of_day', 'is_weekend', 'payment_per_person'
    ]

    # 🔹 Aggregate Data by Hour (if needed)
    if 'bill_total_billed' not in df.columns:
        st.warning("⚠️ Detected raw transaction-level data. Grouping by hour...")

        df_hourly = df.groupby(['venue_xref_id', df['bill_paid_at_local'].dt.floor('H')]).agg({
            'bill_total_net': 'sum',  
            'bill_total_billed': 'sum',  
            'payment_amount': 'sum',  
            'num_people': 'sum',  
            'holiday': 'max',  
            'day_of_week': 'first',  
            'hour_of_day': 'first',  
            'is_weekend': 'first',  
            'payment_per_person': 'mean'  
        }).reset_index()

        # Drop venue column (if model doesn't expect it)
        df_hourly.drop(columns=['venue_xref_id'], inplace=True, errors='ignore')
    
    else:
        df_hourly = df.copy()

    # 🔹 Handle Missing Features
    missing_features = [col for col in expected_features if col not in df_hourly.columns]
    if missing_features:
        st.warning(f"⚠️ Missing features detected: {missing_features}. Filling with default values.")

    for col in missing_features:
        if col in ['holiday', 'day_of_week', 'hour_of_day', 'is_weekend']:
            df_hourly[col] = df_hourly[col].mode()[0] if not df_hourly[col].empty else 0
        else:
            df_hourly[col] = 0  # Assign default numerical value

    # 🔹 Ensure Correct Column Order
    df_hourly = df_hourly[expected_features]

    # 🔹 Check Model's Expected Features
    expected_model_features = ep_model.n_features_in_
    if df_hourly.shape[1] != expected_model_features:
        st.error(f"❌ Feature shape mismatch! Model expects {expected_model_features} features, but received {df_hourly.shape[1]}.")
    else:
        # 🔹 Reshape Test Data If Needed
        X_test = df_hourly.values  # Convert to NumPy array

        if len(X_test.shape) == 1:
            X_test = X_test.reshape(1, -1)  # Ensure correct shape for single-row predictions

        # 🔹 Make Predictions
        df_hourly["predicted_actual_earnings"] = ep_model.predict(X_test)
        df_hourly["predicted_potential_earnings"] = pp_model.predict(X_test)

        # 🔹 Calculate Difference Between Potential and Actual Earnings
        df_hourly["potential_vs_actual"] = df_hourly["predicted_potential_earnings"] - df_hourly["predicted_actual_earnings"]

        # Display Predictions
        st.subheader("📊 Model Predictions")
        st.write(df_hourly[["bill_paid_at_local", "hour_of_day", "predicted_actual_earnings", "predicted_potential_earnings", "potential_vs_actual"]])

        # 🔹 Plot Earnings vs. Potential Earnings Over Time
        st.subheader("📈 Earnings vs. Potential Earnings Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_hourly["hour_of_day"], df_hourly["predicted_actual_earnings"], label="Actual Earnings", marker="o")
        ax.plot(df_hourly["hour_of_day"], df_hourly["predicted_potential_earnings"], label="Potential Earnings", linestyle="dashed", marker="s")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Earnings")
        ax.set_title("Predicted Actual vs. Potential Earnings")
        ax.legend()
        st.pyplot(fig)

        # 🔹 Show Top 5 Underperforming Hours
        st.subheader("⚠️ Top 5 Hours with Highest Missed Revenue")
        top_underperformance = df_hourly.nlargest(5, "potential_vs_actual")[["bill_paid_at_local", "hour_of_day", "predicted_actual_earnings", "predicted_potential_earnings", "potential_vs_actual"]]
        st.write(top_underperformance)

        # 🔹 Feature Importance for Both Models
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
