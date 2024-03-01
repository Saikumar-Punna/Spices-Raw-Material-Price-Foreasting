# -*- coding: utf-8 -*-
"""
Created on Thu Feb 1 06:46:29 2024

@author: sai kumar
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
from joblib import load
from sqlalchemy import create_engine

# Load the models and results
holt_winters_models = load('holt_winters_models.joblib')
forecast_results = load('forecast_results.joblib')

def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    absolute_percentage_error = np.abs(
        (actual - forecast) / actual)
    mape = np.mean(absolute_percentage_error) * 100
    return mape

def connect_to_database(user, pw, db):
    # Create a connection to the MySQL database
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    return engine

def store_results_in_database(engine, table_name, result_df):
    # Store the DataFrame in the specified table in the database
    result_df.to_sql(table_name, con=engine, if_exists='replace', chunksize=1000, index=False)

# Set page config
st.set_page_config(
    page_title="Spices Price Forecasting",
    page_icon="🌶️",
    layout="centered"  # "wide"
)

# Add custom HTML and CSS for light green theme
light_green_theme = """
    <style>
        body {
            background-color: #e6f7e6;  /* Light green background color */
            color: #333;  /* Text color */
        }

        .stApp {
            background-color: #e6f7e6;  /* Light green background color for the entire app */
        }
    </style>
"""
st.markdown(light_green_theme, unsafe_allow_html=True)

# Main heading
st.title("Spices Price Forecasting")

# File uploading and database connection credentials in sidebar
st.sidebar.header('Upload Data and Connect to Database')
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
user = st.sidebar.text_input("Database User", "Type Here")
pw = st.sidebar.text_input("Database Password", "Type Here", type='password')
db = st.sidebar.text_input("Database Name", "Type Here")

if uploaded_file is not None:
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension.lower() == 'csv':
        spices_data = pd.read_csv(uploaded_file)
    elif file_extension.lower() == 'xlsx':
        spices_data = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")

    spices_data['Month&Year'] = pd.to_datetime(spices_data['Month&Year'], format='%d %B %Y', errors='coerce')
    data_copy = spices_data.copy()
    data_copy.drop(["Location", "Grade"], axis=1, inplace=True)

    spices_names = pd.Series(data_copy["Spices"]).unique()
    spice = st.selectbox("Select a spice", spices_names)

    prediction_period = st.slider("Select the prediction period (up to 24 months)", 1, 24, 6)

    if st.button("Predict"):
        if spice and prediction_period:
            spice_data = data_copy[data_copy["Spices"] == spice].copy()
            hw_model = holt_winters_models[spice]['model']
            forecast = hw_model.forecast(steps=prediction_period)
            forecast_values = forecast.values
            actual_values = spice_data['Price'].iloc[-prediction_period:]
            mape = calculate_mape(actual_values, forecast_values)

            # Convert integer index to datetime
            spice_data.index = pd.to_datetime(spice_data['Month&Year'], errors='coerce')

            # Display forecast plot
            st.write(f"### Forecast for {spice} with {prediction_period} months period")
            index = np.arange(len(forecast_values))
            # Convert index to list of strings
            date_strings = [date.strftime('%Y-%m-%d') for date in spice_data.index[-len(index):]]
            plt.plot(date_strings, actual_values, label='Actual')
            plt.plot(date_strings, forecast_values, label='Forecast', color='red')
            plt.title(f'Holt-Winters Forecast for {spice}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

            # Displaying forecasted prices with dates, actual values, forecasted values, and MAPE values
            st.write(f"### Forecasted Prices with Dates, Actual Values, Forecasted Values, and MAPE values")
            # Create initial forecast results dataframe
            forecast_results_df = pd.DataFrame({'Date': date_strings, 'Actual': actual_values, 'Forecast': forecast_values})
            forecast_results_df['MAPE'] = round(mape, 2)

            # Create future dates for the forecasted values
            future_dates = pd.to_datetime(pd.date_range(start='01/1/2024', periods=prediction_period, freq="M"))
            future_forecast = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_values[-prediction_period:]})

            # Concatenate actual and future forecast values
            final_forecast_results_df = pd.concat([forecast_results_df, future_forecast], ignore_index=True)

            # Filter out rows with NaN in the "Actual" column (to show only future predicted values)
            final_forecast_results_df = final_forecast_results_df[final_forecast_results_df['Actual'].notna()]

            # Sort the final_forecast_results_df by the "Date" column
            final_forecast_results_df = final_forecast_results_df.sort_values(by='Date')

            # Displaying forecasted prices with dates and MAPE values
            st.write(f"### Forecasted Prices with Dates and MAPE values")
            st.write(final_forecast_results_df)

            # Displaying machine learning model details
            st.write(f"### Machine Learning Model Details for {spice}")
            st.write(holt_winters_models[spice]['model'])

            # Save results to database
            if st.button("Store Results in Database"):
                # Connect to the database
                engine = connect_to_database(user, pw, db)

                # Specify the table name to store the results
                table_name = "spice_forecast_results"

                # Store the forecast results in the database
                store_results_in_database(engine, table_name, final_forecast_results_df)
                st.success("Results successfully stored in the database.")
