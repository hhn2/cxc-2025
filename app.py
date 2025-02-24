import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from geopy.geocoders import Nominatim
import time
import openmeteo_requests
from retry_requests import retry
import requests

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

ep_model = load_model("ep_model.pkl")  # Earnings Prediction Model
pp_model = load_model("pp_model.pkl")  # Potential Earnings Prediction Model

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

def get_city_coordinates(city, country):
    geolocator = Nominatim(user_agent="weather_app")
    try:
        location = geolocator.geocode(f"{city}, {country}", exactly_one=True, timeout=10)
        if location:
            return (location.latitude, location.longitude)
        else:
            st.warning(f"⚠️ Could not find coordinates for {city}, {country}")
            return None
    except Exception as e:
        st.error(f"❌ Error geocoding {city}, {country}: {e}")
        return None

def get_city_weather_daily(city, country, start_date, end_date):
    coordinates = get_city_coordinates(city, country)
    if not coordinates:
        return None

    lat, lon = coordinates
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                  "relative_humidity_2m_mean", "precipitation_sum"],
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        print("\n🔍 API Response:")
        print(data)  # Debugging the response

        if "daily" not in data or "time" not in data["daily"]:
            st.error(f"❌ Unexpected API response format for {city}, {country}")
            return None

        # Create the DataFrame
        daily_weather = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]).date,
            "city": city,
            "country": country,
            "max_temperature": data["daily"].get("temperature_2m_max", [np.nan] * len(data["daily"]["time"])),
            "min_temperature": data["daily"].get("temperature_2m_min", [np.nan] * len(data["daily"]["time"])),
            "avg_temperature": data["daily"].get("temperature_2m_mean", [np.nan] * len(data["daily"]["time"])),
            "avg_humidity": data["daily"].get("relative_humidity_2m_mean", [np.nan] * len(data["daily"]["time"])),
            "precipitation_sum": data["daily"].get("precipitation_sum", [np.nan] * len(data["daily"]["time"]))
        })

        # Ensure all required columns exist
        for col in ['max_temperature', 'min_temperature', 'avg_temperature', 'avg_humidity', 'precipitation_sum']:
            if col not in daily_weather.columns:
                daily_weather[col] = np.nan  # Create missing columns with NaN

        return daily_weather

    except Exception as e:
        st.error(f"❌ Error fetching weather for {city}, {country}: {e}")
        return None



def get_forecast(city, country, days=7):
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(f"{city}, {country}")

    if not location:
        return None  # Return None if location is not found

    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                  "relative_humidity_2m_mean", "precipitation_sum"],
        "timezone": "auto",
        "forecast_days": days
    }

    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        st.error(f"❌ Error fetching forecast: {response.status_code}")
        return None  # Return None if the request fails

    data = response.json()
    
    if "daily" not in data or "time" not in data["daily"]:
        st.error(f"❌ Unexpected forecast API response format")
        return None

    forecast_df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]).date,
        "city": city,
        "country": country,
        "max_temperature": data["daily"]["temperature_2m_max"],
        "min_temperature": data["daily"]["temperature_2m_min"],
        "avg_temperature": data["daily"]["temperature_2m_mean"],
        "avg_humidity": data["daily"]["relative_humidity_2m_mean"],
        "precipitation_sum": data["daily"]["precipitation_sum"]
    })

    return forecast_df


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
        df['business_date'] = df['bill_paid_at_local'].dt.date

        df['payment_per_person'] = df['payment_amount'] / df['num_people'].replace(0, np.nan)
        df['payment_per_person'] = df['payment_per_person'].fillna(0)  # Replace NaNs with 0
        
        df_hourly = df.groupby(['business_date', df['bill_paid_at_local'].dt.floor('H')]).agg({
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

        start_date = df_hourly['business_date'].min().strftime('%Y-%m-%d')
        end_date = df_hourly['business_date'].max().strftime('%Y-%m-%d')

        if city:
            st.info(f"🌍 Fetching weather data for {city}, {country} ({start_date} - {end_date})...")

            # Fetch historical weather data
            weather_df = get_city_weather_daily(city, country, start_date, end_date)

            # Fetch 7-day weather forecast
            forecast_df = get_forecast(city, country, days=7)

            if weather_df is not None:
                st.success(f"✅ Historical weather data fetched for {city}!")
                df_hourly = df_hourly.merge(
                    weather_df,
                    left_on=['business_date'],
                    right_on=['date'],
                    how='left'
                ).drop(columns=['date'], errors='ignore')

            if forecast_df is not None:
                st.success(f"✅ 7-day weather forecast fetched for {city}!")
                df_hourly = df_hourly.merge(
                    forecast_df,
                    left_on=['business_date'],
                    right_on=['date'],
                    how='left'
                ).drop(columns=['date'], errors='ignore')

            # Ensure all weather columns exist
            for col in ['max_temperature', 'min_temperature', 'avg_temperature', 'avg_humidity', 'precipitation_sum']:
                if col not in df_hourly.columns:
                    df_hourly[col] = np.nan  # Create missing columns with NaN


        
        features = [
            'bill_total_net', 'bill_total_billed', 'payment_amount', 'num_people', 
            'day_of_week', 'hour', 'is_weekend', 'payment_per_person',
            'max_temperature', 'min_temperature', 'avg_temperature', 'avg_humidity', 'precipitation_sum'
        ]

        for col in ['max_temperature', 'min_temperature', 'avg_temperature', 'avg_humidity', 'precipitation_sum']:
            df_hourly[col] = df_hourly[col].fillna(df_hourly[col].mean())

        st.write(df_hourly.head(5))
        
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

