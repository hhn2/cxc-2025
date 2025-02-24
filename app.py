import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="PromoPulse",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("📊 PromoPulse")

@st.cache_resource
def load_model(model_filename):
    with open(model_filename, "rb") as f:
        return pickle.load(f)

ep_model = load_model("ep_model.pkl")  # Earnings Prediction Model
pp_model = load_model("pp_model.pkl")  # Potential Earnings Prediction Model

st.sidebar.title("🔘 Sidebar Controls")
country = st.sidebar.selectbox("Select the Country", options=["US", "CA"])
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

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Uploaded Successfully!")

    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns (or incorrect column naming): {', '.join(missing_columns)}")
    else:
        df['bill_paid_at_local'] = pd.to_datetime(df['bill_paid_at_local'])  # Ensure datetime format
        df['hour'] = df['bill_paid_at_local'].dt.hour  
        df['day_of_week'] = df['bill_paid_at_local'].dt.dayofweek  # (0=Monday, 6=Sunday)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        df['payment_per_person'] = df['payment_amount'] / df['num_people'].replace(0, np.nan)
        df['payment_per_person'] = df['payment_per_person'].fillna(0)  # Replace NaNs with 0
        
        df_hourly = df.groupby([df['bill_paid_at_local'].dt.floor('H')]).agg({
            'bill_total_net': 'sum',  # Sum of net earnings
            'bill_total_billed': 'sum',  # Sum of billed amount
            'bill_total_discount_item_level': 'sum',  # Sum of discounts
            'bill_total_gratuity': 'sum',  # Total gratuity
            'bill_total_tax': 'sum',  # Total tax
            'bill_total_voided': 'sum',  # Sum of voided bills
            'payment_amount': 'sum',  # Total payment amount
            'num_people': 'sum',  # Total number of people
            'payment_total_tip': 'sum',  # Total tips
            'sales_revenue_with_tax': 'sum',  # Total revenue including tax
            'is_weekend': 'first',  # First value of is_weekend per hour
            'day_of_week': 'first',  # First value of day_of_week per hour
            'hour': 'first',  # First value of hour per hour
            'payment_per_person': 'mean'  # Mean payment per person for that hour
        }).reset_index()
        
        st.success("✅ Aggregated Data to Hourly Level Successfully!")
        
        features = ['bill_total_net', 'bill_total_billed', 'payment_amount', 'num_people', 
                    'day_of_week', 'hour', 'is_weekend', 'payment_per_person']
        
        df_hourly['predicted_actual_earnings'] = ep_model.predict(df_hourly[features])
        df_hourly['predicted_potential_earnings'] = pp_model.predict(df_hourly[features])
        df_hourly['potential_vs_actual'] = df_hourly['predicted_potential_earnings'] - df_hourly['predicted_actual_earnings']
        
        st.success("✅ Predictions Made Successfully!")
        
        st.subheader("📊 Aggregated Hourly Data with Predictions")
        st.write(df_hourly[['bill_paid_at_local', 'hour', 'predicted_actual_earnings', 'predicted_potential_earnings', 'potential_vs_actual']])

        st.subheader("📈 Actual Earnings vs. Potential Earnings")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_hourly['hour'], df_hourly['predicted_actual_earnings'], label="Predicted Actual Earnings", marker="o")
        ax.plot(df_hourly['hour'], df_hourly['predicted_potential_earnings'], label="Predicted Potential Earnings", linestyle="dashed", marker="s")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Earnings")
        ax.set_title("Predicted Actual vs. Potential Earnings")
        ax.legend()
        st.pyplot(fig)

