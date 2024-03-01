# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 08:26:31 2024

@author: sai kumar
"""
'''
Business Problem: 
The business problem at hand is the unpredictable fluctuation in the prices of raw spice materials, 
negatively impacting the cost structure and inventory management.

Business objective:
Maximize cost savings through effective inventory management.
Business Constraint:
Minimize the impact of price volatility on production costs and optimize 
procurement strategies to ensure stable and affordable raw material sourcing.

Success criteria:
    
Business success criteria:
To optimize procurement strategies and reduced production costs by 10%
ML success criteria:
Achieve an accuracy of a least 95%
Economic success criteria:
To achieve cost savings in raw material procurement and inventory management at least by 20%
'''

# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from scipy.stats import iqr, skew, kurtosis
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import kpss
import joblib
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine

# Connect to the MySQL database
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="user2", pw="4556", db="Spices"))

# Load data into a Pandas DataFrame
data = pd.read_csv(r"C:\Users\sai kumar\OneDrive\Documents\My Project\Model Building\Spices_Data.csv")
data.info()
data.isnull().sum()

print(data['Month&Year'].unique())

# Find the starting date and ending date
start_date = data['Month&Year'].min()
end_date = data['Month&Year'].max()

print("Starting Date:", start_date)
print("Ending Date:", end_date)

# Convert 'Date' column to datetime format
data['Month&Year'] = pd.to_datetime(data['Month&Year'], format='%d %B %Y', errors='coerce')

print(data)

# Dump data into the database
data.to_sql('data', con=engine, if_exists='replace', index=False, chunksize=1000)

# Load data from the database
sql = 'SELECT * FROM data'
data = pd.read_sql_query(sql, engine)

#Data Information
data.info()

data.describe()

# Create a copy of the DataFrame after loading from the database
data_copy = data.copy()

# Drop unwanted columns
data_copy.drop(["Location", "Grade"], axis=1, inplace=True)

data_copy.info()

# Extract unique spice names
spices_names = pd.Series(data_copy["Spices"]).unique()

# Initialize the spices_dataframes dictionary before the loop
spices_dataframes = {}

# Loop through each spice
for spice in spices_names:
    # Filter the data set for each spice
    filtered_data = data_copy[data_copy["Spices"] == spice].copy()
    
    # Store the filtered dataframe in the dictionary with spice names as the key
    spices_dataframes[spice] = filtered_data

# Create a KNNImputer
imputer = KNNImputer(n_neighbors=5)


# Loop through each spice
for spice in spices_names:
    # Filter the data set for each spice
    filtered_data = data_copy[data_copy["Spices"] == spice].copy()
    
    # Extract relevant columns for imputation
    impute_data = filtered_data[["Price"]]
    
    # Perform kNN imputation on the Price column
    imputed_values = imputer.fit_transform(impute_data)
    
    # Assign the imputed values back to the original DataFrame
    filtered_data.loc[:, "Price"] = imputed_values
    
    # Store the imputed DataFrame in the dictionary with spice names as the key
    spices_dataframes[spice] = filtered_data
    
    # Extract rows with NaN values in the "Price" column after imputation
    nan_rows_after = spices_dataframes[spice][spices_dataframes[spice]["Price"].isna()]
    print(f"NaN values in {spice} data frame after imputation:")
    print(nan_rows_after)
    
joblib.dump(imputer, 'imputer.joblib')

spice_statistics = {}

# Loop through each spice and calculate statistics
for spice, spice_data in spices_dataframes.items():
    # Check if there are enough data points to calculate statistics
    if len(spice_data) > 0:
        # Measures of Central Tendency
        mean_price = spice_data['Price'].mean()
        median_price = spice_data['Price'].median()
        
        # Handle mode when multiple modes exist
        mode_price = spice_data['Price'].mode()
        if not mode_price.empty:
            mode_price = mode_price.iloc[0]
        
        # Measures of Dispersion
        min_price = spice_data['Price'].min()
        max_price = spice_data['Price'].max()
        q1_price = spice_data['Price'].quantile(0.25)
        q3_price = spice_data['Price'].quantile(0.75)
        iqr_price = q3_price - q1_price
        std_dev_price = spice_data['Price'].std()
        var_price = spice_data['Price'].var()
        
        # Skewness and Kurtosis
        price_skewness = skew(spice_data['Price'])
        price_kurtosis = kurtosis(spice_data['Price'])
        
        # Store statistics in a dictionary for each spice
        spice_statistics[spice] = {
            'Mean': mean_price,
            'Median': median_price,
            'Mode': mode_price,
            'Min': min_price,
            'Max': max_price,
            'Q1': q1_price,
            'Q3': q3_price,
            'IQR': iqr_price,
            'Std Dev': std_dev_price,
            'Var': var_price,
            'Skewness': price_skewness,
            'Kurtosis': price_kurtosis
        }
    else:
        print(f"Not enough data points for statistics calculation for {spice}.")
else:
    print(f"No data available for {spice}.")
        
        

# Display the statistics for each spice
for spice, stats in spice_statistics.items():
    print(f"\nStatistics for {spice}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")

# EDA for all spices
grouped_final = data_copy.groupby("Spices")

num_rows = 3
num_cols = (len(grouped_final) + num_rows - 1) // num_rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

for (name, group), ax in zip(grouped_final, axes.flatten()):
    group["Price"].plot(kind="box", ax=ax)
    ax.set_title(name)
    ax.set_ylabel("Price")
    ax.grid(True)

# Hide any extra empty subplots
for i in range(len(grouped_final), num_rows * num_cols):
    axes.flatten()[i].axis("off")

plt.suptitle("Price Distribution of spices (after Winsorization using IQR)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# List of spices with outliers
outlier_spices = ["AJWAN SEED", "CARDAMOM(SMALL)", "CASSIA", "CLOVE", "FENUGREEK", "MUSTARD", "SAFFRON"]

winsorized_data_list = []

# Loop through each group
for name, group in grouped_final:
    # Impute missing values for the 'Price' column using mean
    imputer = SimpleImputer(strategy='median')
    group['Price'] = imputer.fit_transform(group[['Price']])
    
    # Check if the spice has outliers
    if name in outlier_spices:
        # Create a Winsorizer instance
        winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Price'])
        
        # Fit and transform the 'Price' column
        group['Price'] = winsorizer.fit_transform(group[['Price']])
    
    # Append the group to the list
    winsorized_data_list.append(group)

# Concatenate the winsorized dataframes into a single DataFrame
winsorized_data = pd.concat(winsorized_data_list, ignore_index=True)

# Treating outliers
# Create subplots for individual spices
num_rows = 3
num_cols = (len(spices_dataframes) + num_rows - 1) // num_rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Loop through each spice and plot box plots
for i, (spice, spice_data) in enumerate(spices_dataframes.items()):
    # Check if there are samples in the 'Price' column
    if not spice_data['Price'].empty:
        
        # Check if the spice has outliers
        if spice in outlier_spices:
            # Create a Winsorizer instance
            winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Price'])
            
            # Fit and transform the 'Price' column
            spice_data['Price'] = winsorizer.fit_transform(spice_data[['Price']])
        
        # Box plot
        spice_data.boxplot(column='Price', ax=axes[i])
        axes[i].set_title(spice)
        axes[i].set_ylabel("Price")
        axes[i].grid(True)
        
# Hide any extra empty subplots
for j in range(len(spices_dataframes), num_rows * num_cols):
    axes[j].axis("off")

# Adjust layout
plt.suptitle("Price Distribution of Spices (after winsorization)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

joblib.dump(winsorizer, 'winsorizer.joblib')

# Check for missing values in the "None" time series
if data_copy[data_copy['Spices'] == 'None']['Price'].isnull().any():
    # Impute missing values using forward fill (you can choose another imputation method if needed)
    data_copy.loc[data_copy['Spices'] == 'None', 'Price'].fillna(method='ffill', inplace=True)

# Serialize preprocessed data
joblib.dump(data_copy, 'preprocessed_data.joblib')  

        ############ Auto EDA ################
        
import sweetviz as sv
s = sv.analyze(data_copy)
s.show_html()

import dtale
d = dtale.show(data_copy)
d.open_browser()


'''
 decompose_and_random_test(spice, spice_data):
 Purpose: Performs seasonal decomposition and random testing on the time series data for a specific spice.
 Usage: Investigates the underlying patterns, seasonality, and randomness in spice prices.
 '''

# Function to perform seasonal decomposition and random test for each spice
def decompose_and_random_test(spice, spice_data):
    if spice_data['Price'].dropna().empty:
        print(f"The time series for {spice} is empty or contains only NaN values.")
        return
    
    # Time series decomposition
    period = min(6, len(spice_data['Price']) - 1)
    decomposition = sm.tsa.seasonal_decompose(spice_data['Price'], model='multiplicative', period=period)
    
    # Plot decomposition components
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    decomposition.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    decomposition.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    decomposition.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    plt.suptitle(f"Seasonal Decomposition for {spice}")
    
    # Random walk check using AutoRegressive (AR) model
    try:
        model = sm.tsa.AutoReg(spice_data['Price'], lags=6)
        result = model.fit()
        autocorr_coefficient = result.params[1]
        if autocorr_coefficient > 0.95:  # Check if the autoregressive coefficient is close to 1
            print(f"The time series for {spice} has characteristics of a random walk.")
        else:
            print(f"The time series for {spice} does not appear to be a random walk.")
    except NotImplementedError:
        print("AutoReg is not implemented in your version of statsmodels.")
    
    plt.show()

# Iterating through spice data
for spice, spice_data in spices_dataframes.items():
    decompose_and_random_test(spice, spice_data)
    
'''After checking for a random walk using an AutoRegressive (AR) model,
 the analysis confirms that there is no evidence of a random walk in the data.
 '''
 
# Function to check stationarity of a time series using ADF and KPSS tests
def check_stationarity(series, name):
    if series.dropna().empty:
        print(f"The time series for {name} is empty or contains only NaN values.")
        return

    # ADF test
    result_adf = adfuller(series.dropna())
    print(f"ADF Test for {name}:")
    print(f"ADF Statistic: {result_adf[0]}")
    print(f"P-value: {result_adf[1]}")
    print(f"Critical Values: {result_adf[4]}")
    if result_adf[1] <= 0.05:
        print(f"The time series for {name} is likely stationary.")
    else:
        print(f"The time series for {name} is likely non-stationary.")

    # KPSS test
    result_kpss = kpss(series)
    print(f"\nKPSS Test for {name}:")
    print(f"KPSS Statistic: {result_kpss[0]}")
    print(f"P-value: {result_kpss[1]}")
    print(f"Critical Values: {result_kpss[3]}")
    if result_kpss[1] >= 0.05:
        print(f"The time series for {name} is likely non-stationary according to KPSS.")
    else:
        print(f"The time series for {name} is likely stationary according to KPSS.")

# Loop through each spice and check stationarity using ADF and KPSS tests
for spice, spice_data in spices_dataframes.items():
    check_stationarity(spice_data['Price'], spice)

# Function to plot ACF and PACF
def plot_acf_pacf(series, spice_name):
    num_lags_acf = 12  # Set the number of lags for ACF to 12
    num_lags_pacf = 12  # Set the number of lags for PACF to 12

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    sm.graphics.tsa.plot_acf(series, lags=num_lags_acf, ax=ax1)
    plt.title(f'ACF for {spice_name}')

    ax2 = plt.subplot(2, 1, 2)
    try:
        sm.graphics.tsa.plot_pacf(series, lags=num_lags_pacf, ax=ax2)
    except ValueError as e:
        if "Can only compute partial correlations for lags up to 50% of the sample size" in str(e):
            plt.text(0.5, 0.5, "PACF not calculated due to insufficient data", ha="center", va="center")
        else:
            raise e
    plt.title(f'PACF for {spice_name}')

    plt.tight_layout()
    plt.show()

# Loop through each spice and plot ACF and PACF
for spice, spice_data in spices_dataframes.items():
    plot_acf_pacf(spice_data['Price'], spice)

# Function to check trend, seasonality, and residuals
def check_trend_seasonality_resid(spice, spice_data):
    if spice_data['Price'].dropna().empty:
        print(f"The time series for {spice} is empty or contains only NaN values.")
        return
    
    # Perform time series decomposition with a reduced period
    period = min(6, len(spice_data['Price']) - 1)
    decomposition = seasonal_decompose(spice_data['Price'], model='multiplicative', period=period)


    # Plot the original time series
    plt.figure(figsize=(15, 5))
    plt.plot(spice_data['Month&Year'], spice_data['Price'], label='Original Time Series')
    plt.title(f"Original Time Series for {spice}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plot the trend, seasonality, and residuals
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(decomposition.trend, label='Trend')
    plt.title(f"Trend for {spice}")
    plt.xlabel('Date')
    plt.ylabel('Trend')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.title(f"Seasonality for {spice}")
    plt.xlabel('Date')
    plt.ylabel('Seasonality')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(decomposition.resid, label='Residuals')
    plt.title(f"Residuals for {spice}")
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Loop through each spice and perform decomposition and trend/seasonality/residuals check
for spice, spice_data in spices_dataframes.items():
    decompose_and_random_test(spice, spice_data)
    check_stationarity(spice_data['Price'], spice)
    if not spice_data['Price'].dropna().empty:
        plot_acf_pacf(spice_data['Price'], spice)
        check_trend_seasonality_resid(spice, spice_data)

# Function to difference the time series
def difference_series(series, lag=6):
    return series.diff(lag).dropna()

# Function to invert differencing
def invert_difference(initial_value, diff_series, lag=6):
    cumsum = diff_series.cumsum()
    return initial_value + cumsum.shift(-lag).fillna(0)

# Loop through each spice, difference the time series, and check stationarity again
for spice, spice_data in spices_dataframes.items():
    diff_price = difference_series(spice_data['Price'])
    check_stationarity(diff_price, f"{spice} (Differenced)")

    # Optionally, visualize the differenced time series
    plt.figure(figsize=(12, 6))
    plt.plot(diff_price)
    plt.title(f"Differenced Time Series for {spice}")
    plt.xlabel("Date")
    plt.ylabel("Differenced Price")
    plt.show()

# Function to check stationarity using KPSS test
def check_stationarity_kpss(series, name):
    # Print information about the time series before handling missing values
    print(f"Before handling missing values for {name}:")
    print(series.head())
    
    # Drop missing values from the series before performing the test
    series = series.dropna()
    
    # Print information about the time series after handling missing values
    print(f"After handling missing values for {name}:")
    print(series.head())
    
    if series.empty:
        print(f"The time series for {name} is empty or contains only NaN values.")
        return

    result = kpss(series)
    print(f"KPSS Test for {name}:")
    print(f"KPSS Statistic: {result[0]}")
    print(f"P-value: {result[1]}")
    print(f"Critical Values: {result[3]}")
    if result[1] >= 0.05:
        print(f"The time series for {name} is likely non-stationary according to KPSS.")
    else:
        print(f"The time series for {name} is likely stationary according to KPSS.")

# Loop through each spice and check stationarity using KPSS test
for spice, spice_data in spices_dataframes.items():
    check_stationarity_kpss(spice_data['Price'], spice)
    
    
'''
After Defferncing Method There are only three spices with 
non-stationary. 
They are: [TURMERIC, CLOVE, CARDAMOM(SMALL)]
'''
###After Observing all Transformation Method I decided that Defferncing method is working well
##Defferencing method To Make Time series Stationary:
#Calculate the difference between consecutive observations.

import numpy as np
def check_stationarity(series):
    # Drop NaN values from the series
    series = series.dropna()
    
    # ADF test
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:', result[4])
    
    # Check stationarity
    if result[1] <= 0.05:
        print("ADF test: Series is stationary (reject the null hypothesis)")
    else:
        print("ADF test: Series is non-stationary (fail to reject the null hypothesis)")

# Check stationarity after differencing
for spice_name, spice_data in spices_dataframes.items():
    # Calculate price difference and drop NaN values
    spice_data['Price_Diff'] = spice_data['Price'].diff().dropna()
    
    # Print spice name
    print(f"Checking stationarity for {spice_name} after differencing:")
    
    # Print the first few rows of the differenced series
    print(spice_data['Price_Diff'].head())
    
    # Check stationarity after differencing
    check_stationarity(spice_data['Price_Diff'])
    
    print("\n" + "="*50 + "\n")
    
#######################Holt-Winters Model###############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)

    # Calculate absolute percentage error
    absolute_percentage_error = np.abs(
        (actual - forecast) / actual)

    # Calculate mean absolute percentage error
    mape = np.mean(absolute_percentage_error) * 100

    return mape

holt_winters_results = {}

for spice_name, spice_data in spices_dataframes.items():
    try:
        price_series = spice_data['Price']

        train_data = price_series.loc[(spice_data['Month&Year'] >= '2021-04-01') & (spice_data['Month&Year'] < '2023-01-01')]
        validation_data = price_series.loc[(spice_data['Month&Year'] >= '2023-01-01') & (spice_data['Month&Year'] <= '2023-12-01')]

        best_mape = float('inf')
        best_hw_model = None

        for trend_type in ['add', 'multiplicative']:
            for seasonal_type in ['add', 'multiplicative']:
                for seasonal_period in [4]:
                    try:
                        hw_model = ExponentialSmoothing(
                            train_data,
                            trend=trend_type,
                            seasonal=seasonal_type,
                            seasonal_periods=seasonal_period,
                            initialization_method="estimated"
                        )

                        hw_fit = hw_model.fit()

                        hw_forecast = hw_fit.forecast(steps=12)

                        mape = calculate_mape(validation_data, hw_forecast)

                        if mape < best_mape:
                            best_mape = mape
                            best_hw_model = hw_fit

                    except ValueError as ve:
                        print(f"Error building Holt-Winters model for {spice_name}: {ve}")
                        continue  # Continue to the next iteration in case of an error
                    except Exception as e:
                        print(f"Error building Holt-Winters model for {spice_name}: {e}")
                        continue

        if best_hw_model is not None:
            holt_winters_results[spice_name] = {'model': best_hw_model, 'mape': best_mape}

    except Exception as general_exception:
        print(f"General error for {spice_name}: {general_exception}")
    
    if best_hw_model is not None:
        # Store the best model in the dictionary
        holt_winters_results[spice_name] = {
            'model': best_hw_model, 'mape': best_mape}

# Loop through each spice and print RMSE and MAPE
for spice_name, result in holt_winters_results.items():
    hw_model = result['model']
    actual_values = spices_dataframes[spice_name]['Price']
    
    # Make predictions on the entire dataset
    hw_forecast = hw_model.fittedvalues
    
    # Check and align data lengths
    if len(actual_values) != len(hw_forecast):
        # Adjust the length of actual_values or hw_forecast to make them consistent
        min_len = min(len(actual_values), len(hw_forecast))
        actual_values = actual_values[:min_len]
        hw_forecast = hw_forecast[:min_len]
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(actual_values, hw_forecast))
    
    # Calculate MAPE
    mape = calculate_mape(actual_values, hw_forecast)
    
    print(f"\nResults for {spice_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot actual vs. predicted values
    # Use the forecasted time index for plotting
    plt.plot(spices_dataframes[spice_name].index[:len(hw_forecast)], actual_values, label='Actual')
    plt.plot(spices_dataframes[spice_name].index[:len(hw_forecast)], hw_forecast, label='Holt-Winters Forecast', color='red')
    plt.title(f'Holt-Winters Forecast for {spice_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Generate forecasts for future periods
forecast_results = {}
confidence_level = 0.95
forecast_periods = 12  # Replace with your desired number of periods

for spice_name, result in holt_winters_results.items():
    hw_model = result['model']

    # Use forecast method
    forecast = hw_model.forecast(steps=forecast_periods)
    forecast_values = forecast.values

    # Calculate standard errors and confidence intervals (adjust based on available error terms)
    if hasattr(hw_model, 'resid'):  # Check for available residuals
        sigma2 = np.mean(hw_model.resid**2)  # Use model residuals if available
    else:
        sigma2 = np.mean((forecast_values - spices_dataframes[spice_name]['Price'].iloc[-forecast_periods:])**2)  # Recalculate

    # Adjust std_errors calculation for missing errors attribute
    std_errors = np.sqrt(sigma2 * np.ones(len(forecast)))  # Assume equal standard errors

    lower_ci = forecast_values - np.percentile(std_errors, (1 - confidence_level) * 100, axis=0)
    upper_ci = forecast_values + np.percentile(std_errors, (1 - confidence_level) * 100, axis=0)

    # Handle potential missing index
    try:
        index = forecast_values.index
    except AttributeError:
        index = range(len(forecast_values))

    forecast_results[spice_name] = {
        'forecast': forecast_values,
        'confidence_interval': {'lower': lower_ci, 'upper': upper_ci}
    }

    # Plot forecast and confidence interval (align data lengths)
    plt.plot(index, spices_dataframes[spice_name]['Price'].iloc[-len(index):], label='Actual')  # Plot only last 12 values
    plt.plot(index, forecast_values, label='Forecast', color='red')
    plt.fill_between(index, lower_ci, upper_ci, color='red', alpha=0.2, label='Confidence Interval')
    plt.title(f'Holt-Winters Forecast for {spice_name}')
    plt.xlabel('Date' if isinstance(index, pd.DatetimeIndex) else 'Index')  # Adjust label for index type
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Analyze the forecast_results dictionary to understand predicted price dynamics
for spice_name, result in forecast_results.items():
    forecast_values = result['forecast']
    confidence_interval = result['confidence_interval']

    # Print or visualize the forecast values and confidence intervals
    print(f"\nForecast for {spice_name}:")
    print(forecast_values)
    print("\nConfidence Interval:")
    print(f"Lower Bound: {confidence_interval['lower']}")
    print(f"Upper Bound: {confidence_interval['upper']}")

# Save the models and related data
joblib.dump(holt_winters_results, 'holt_winters_models.joblib')
joblib.dump(forecast_results, 'forecast_results.joblib')
print(spice_data['Month&Year'])
