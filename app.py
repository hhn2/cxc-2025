import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

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

def preprocess_input(df):
    """Prepares venue-level sequences for model inference."""

    # Ensure datetime format
    df['bill_paid_at_local'] = pd.to_datetime(df['bill_paid_at_local'])
    df = df.sort_values(by="bill_paid_at_local").reset_index(drop=True)

    # Group data at an hourly level
    df_hourly = df.groupby(df['bill_paid_at_local'].dt.floor('h')).agg({
        'bill_total_net': 'sum',
        'bill_total_billed': 'sum',
        'bill_total_discount_item_level': 'sum',
        'bill_total_gratuity': 'sum',
        'bill_total_tax': 'sum',
        'bill_total_voided': 'sum',
        'payment_amount': 'sum',
        'num_people': 'sum',
        'payment_total_tip': 'sum',
        'sales_revenue_with_tax': 'sum',
        'holiday': 'first',
        'is_weekend': 'first',
        'day_of_week': 'first',
        'hour_of_day': 'first',
        'payment_per_person': 'mean'
    }).reset_index()

    features = [
        'bill_total_net', 'bill_total_billed', 'bill_total_discount_item_level',
        'bill_total_gratuity', 'bill_total_tax', 'bill_total_voided',
        'payment_amount', 'num_people', 'payment_total_tip', 'sales_revenue_with_tax',
        'holiday', 'day_of_week', 'hour_of_day', 
        'is_weekend', 'payment_per_person'
    ]

    # Apply saved scaler
    df_hourly[features] = scaler.transform(df_hourly[features])

    # ✅ Match expected sequence length
    required_hours = 2304  # Model was trained on sequences of this length

    # If data has fewer hours than required, pad with zeros
    if df_hourly.shape[0] < required_hours:
        pad_rows = required_hours - df_hourly.shape[0]
        pad_df = pd.DataFrame(np.zeros((pad_rows, len(features))), columns=features)
        df_hourly = pd.concat([pad_df, df_hourly], ignore_index=True)

    # Keep only the most recent 2304 rows
    df_hourly = df_hourly.iloc[-required_hours:]

    # ✅ Preserve correct shape: (sequence_length, num_features)
    X_hourly = df_hourly[features].values  # (2304, 15)

    # ✅ Flattened version for `ep_model`
    X_flattened = X_hourly.flatten().reshape(1, -1)  # (1, 34560)

    return X_hourly, X_flattened, df_hourly


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if df.shape[0] == 0:
            st.error("❌ The uploaded CSV has headers but no data. Please upload a valid file.")
        else:
            st.success("✅ Data Uploaded Successfully!")

            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            else:
                # ✅ Prepare inputs
                X_hourly, X_flattened, df_hourly = preprocess_input(df)

                # **Ensure correct feature count**
                expected_features_ep = ep_model.get_booster().num_features()
                expected_features_pp = pp_model.get_booster().num_features()

                st.write(f"📏 `ep_model` expects {expected_features_ep} features")
                st.write(f"📏 `pp_model` expects {expected_features_pp} features")

                # ✅ Apply correct shape to each model
                if expected_features_ep == 15:
                    df_hourly['predicted_actual_earnings'] = ep_model.predict(X_hourly)
                else:
                    df_hourly['predicted_actual_earnings'] = ep_model.predict(X_flattened)[0]

                if expected_features_pp == 15:
                    df_hourly['predicted_potential_earnings'] = pp_model.predict(X_hourly)
                else:
                    df_hourly['predicted_potential_earnings'] = pp_model.predict(X_flattened)[0]

                df_hourly['potential_vs_actual'] = df_hourly['predicted_potential_earnings'] - df_hourly['predicted_actual_earnings']

                st.success("✅ Predictions Made Successfully!")

                # ✅ Display results
                st.subheader("📊 Aggregated Hourly Data with Predictions")
                st.write(df_hourly[['bill_paid_at_local', 'hour_of_day', 'predicted_actual_earnings', 'predicted_potential_earnings', 'potential_vs_actual']])

                # ✅ Plot results
                st.subheader("📈 Actual Earnings vs. Potential Earnings")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_hourly['hour_of_day'], df_hourly['predicted_actual_earnings'], label="Predicted Actual Earnings", marker="o")
                ax.plot(df_hourly['hour_of_day'], df_hourly['predicted_potential_earnings'], label="Predicted Potential Earnings", linestyle="dashed", marker="s")
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Earnings")
                ax.set_title("Predicted Actual vs. Potential Earnings")
                ax.legend()
                st.pyplot(fig)

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file appears to be empty. Please upload a valid CSV file.")
    except UnicodeDecodeError:
        st.error("❌ Encoding error. Try re-saving the CSV file with UTF-8 encoding.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
