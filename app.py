import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

st.set_page_config(
    page_title="PromoPulse",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("💸 PromoPulse")

@st.cache_resource
def load_model(model_filename):
    with open(model_filename, "rb") as f:
        return pickle.load(f)

ep_model = load_model("ep_model0.pkl")  # Earnings Prediction Model
pp_model = load_model("pp_model0.pkl")  # Potential Earnings Prediction Model

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.sidebar.title("*️⃣ Controls")
country = st.sidebar.selectbox("Select the Country", options=["US", "CA"])
city = st.sidebar.text_input("Type in a City")
st.sidebar.caption("Make sure the city spelling is correct. ")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
st.sidebar.markdown("Upload your restaurant's bill data (CSV) above. Ensure that you have the following data and match the names: ")
st.sidebar.markdown("- bill_paid_at_local: Datetime when the bill was paid, in the venue's local timezone.")
st.sidebar.markdown("- bill_total_net: Net sales amount (post-discount), excluding tax or gratuity.")
st.sidebar.markdown("- bill_total_billed: Total amount billed, including tax.")
st.sidebar.markdown("- bill_total_discount_item_level: Total discount amount applied to the bill.")
st.sidebar.markdown("- bill_total_gratuity: Total gratuity paid for the bill.")
st.sidebar.markdown("- bill_total_tax: Total tax amount paid for the bill.")
st.sidebar.markdown("- bill_total_voided: Total voided amount for the bill.")
st.sidebar.markdown("- payment_amount: Total amount paid for the bill.")
st.sidebar.markdown("- num_people: Number of payments made for the bill.")
st.sidebar.markdown("- payment_total_tip: Total tip amount paid.")
st.sidebar.markdown("- sales_revenue_with_tax: Total sales amount (including tax) as displayed on the bill, excluding gratuity.")
st.sidebar.markdown("❗️ Missing data will result in less accurate results. ")

required_columns = [
    'bill_paid_at_local', 'bill_total_net', 'bill_total_billed', 'bill_total_discount_item_level',
    'bill_total_gratuity', 'bill_total_tax', 'bill_total_voided', 'payment_amount', 'num_people',
    'payment_total_tip', 'sales_revenue_with_tax'
]

features = [
        'bill_total_net', 'bill_total_billed', 'bill_total_discount_item_level',
        'bill_total_gratuity', 'bill_total_tax', 'bill_total_voided',
        'payment_amount', 'num_people', 'payment_total_tip', 'sales_revenue_with_tax',
        'holiday', 'day_of_week', 'hour_of_day', 
        'is_weekend', 'payment_per_person'
    ]

def preprocess_input(df):
    """Prepares venue-level sequences for model inference."""
    df['bill_paid_at_local'] = pd.to_datetime(df['bill_paid_at_local'])
    df = df.sort_values(by="bill_paid_at_local").reset_index(drop=True)

    df_hourly = df.groupby(df['bill_paid_at_local'].dt.floor('h')).agg({
        'bill_total_net': 'sum', 'bill_total_billed': 'sum', 
        'bill_total_discount_item_level': 'sum', 'bill_total_gratuity': 'sum',
        'bill_total_tax': 'sum', 'bill_total_voided': 'sum', 
        'payment_amount': 'sum', 'num_people': 'sum', 
        'payment_total_tip': 'sum', 'sales_revenue_with_tax': 'sum',
        'holiday': 'first', 'is_weekend': 'first',
        'day_of_week': 'first', 'hour_of_day': 'first',
        'payment_per_person': 'mean'
    }).reset_index()

    # Normalize using saved scaler
    df_hourly[features] = scaler.transform(df_hourly[features])

    # Ensure input shape consistency (last 2304 hours only)
    required_hours = 2304
    if df_hourly.shape[0] < required_hours:
        pad_rows = required_hours - df_hourly.shape[0]
        pad_df = pd.DataFrame(np.zeros((pad_rows, len(features))), columns=features)
        df_hourly = pd.concat([pad_df, df_hourly], ignore_index=True)

    df_hourly = df_hourly.iloc[-required_hours:]  # Keep only the last 2304 hours

    # ✅ Start predictions from **TODAY**
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    # Generate timestamps for the next 168 hours
    future_timestamps = pd.date_range(start=today, periods=168, freq='H')

    # ✅ Create a future DataFrame with timestamps & correct time-related features
    future_df = pd.DataFrame({'bill_paid_at_local': future_timestamps})
    future_df['hour_of_day'] = future_df['bill_paid_at_local'].dt.hour
    future_df['day_of_week'] = future_df['bill_paid_at_local'].dt.dayofweek
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['holiday'] = 0  # Default assumption: no holidays
    future_df[features] = 0  # Placeholder values

    # ✅ Extract the last week's hourly revenue pattern
    past_week_data = df_hourly.iloc[-168:]  # Get the last 168 hours
    hourly_pattern = past_week_data['bill_total_net'] / past_week_data['bill_total_net'].sum()  # Normalize

    return df_hourly, future_df, future_df[features].values, hourly_pattern


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if df.shape[0] == 0:
            st.error("❌ The uploaded CSV has headers but no data. Please upload a valid file.")
        else:
            st.success("✅ Data Uploaded Successfully!")

            df_hourly, future_df, X_future, hourly_pattern = preprocess_input(df)

            expected_features_ep = ep_model.get_booster().num_features()
            expected_features_pp = pp_model.get_booster().num_features()

            # ✅ Predict **total earnings for the next week**
            predicted_weekly_total = ep_model.predict(X_future)[0]  # Single value

            # ✅ Redistribute into hourly predictions based on historical pattern
            future_df['predicted_actual_earnings'] = predicted_weekly_total * hourly_pattern.values

            # ✅ Predict with `pp_model` for each hourly row
            future_df['predicted_potential_earnings'] = [
                pp_model.predict(X_future[i].reshape(1, -1))[0] for i in range(X_future.shape[0])
            ]

            predicted_weekly_total = ep_model.predict(X_future)[0]  # Single value

            # Compute difference
            future_df['potential_vs_actual'] = future_df['predicted_potential_earnings'] - future_df['predicted_actual_earnings']

            st.success("✅ Predictions Made Successfully!")

            # ✅ Display results
            st.subheader("📊 Forecasted Hourly Data for Next 7 Days")
            st.write(future_df[['bill_paid_at_local', 'hour_of_day', 'predicted_actual_earnings', 'predicted_potential_earnings', 'potential_vs_actual']])

            # ✅ Plot results
            st.subheader("📈 Forecasted Earnings for Next 7 Days")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(future_df['bill_paid_at_local'], future_df['predicted_actual_earnings'], label="Predicted Actual Earnings", marker="o")
            ax.plot(future_df['bill_paid_at_local'], future_df['predicted_potential_earnings'], label="Predicted Potential Earnings", linestyle="dashed", marker="s")
            ax.set_xlabel("Date & Time")
            ax.set_ylabel("Earnings")
            ax.set_title("Predicted Actual vs. Potential Earnings (Next 7 Days)")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ✅ Get Top 5 Hours with the Biggest Difference
            st.subheader("🔥 Top 5 Hours with Highest Revenue Gap")
            top_5_hours = future_df.nlargest(5, 'potential_vs_actual')
            st.write(top_5_hours[['bill_paid_at_local', 'potential_vs_actual']])

            # ✅ Bar Chart for Top 5 Hours
            st.subheader("📊 Top 5 Hours with Highest Revenue Gap")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top_5_hours['bill_paid_at_local'].astype(str), top_5_hours['potential_vs_actual'], color='red')
            ax.set_xlabel("Revenue Difference")
            ax.set_ylabel("Timestamp")
            ax.set_title("Top 5 Hours with Revenue Difference")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

