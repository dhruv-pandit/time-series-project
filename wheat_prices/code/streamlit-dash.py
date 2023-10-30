import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
from statsmodels.graphics.tsaplots import acf, pacf
import plotly.graph_objects as go
from prophet import Prophet

# Read data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/dhruv-pandit/time-series-project/main/wheat_prices/Datasets/PWHEAMTUSDM.xls'
    df_wheat = pd.read_excel(url, skiprows=10, engine='xlrd').rename(columns={'PWHEAMTUSDM' : 'Wheat_Price'})
    return df_wheat

# Check stationarity
def check_stationarity(series):
    test_results = adfuller(series)
    if test_results[1] <= 0.05:
        return 'Series is Stationary'
    else:
        return 'Series is Non Stationary'

# App title and introduction
st.title("Time Series Analysis of Wheat Prices")
st.write("""
## SARIMAX and Prophet Models
* Contact at dhruvpandit@aln.iseg.ulisboa.pt
""")

# Load and display data
df_wheat = load_data()
st.write("Here's a glance at the data for wheat prices given in USD.")
st.write(df_wheat.head())

# Interactive time series plot of wheat prices
st.subheader("Interactive Plot of Wheat Prices")

# Allow users to select a date range
start_date = st.date_input("Start date", df_wheat['observation_date'].min().date())
end_date = st.date_input("End date", df_wheat['observation_date'].max().date())
if start_date > end_date:
    st.warning("End date should fall after start date.")

# Convert the date objects to datetime64[ns]
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Filter data based on selected date range
mask = (df_wheat['observation_date'] >= start_date) & (df_wheat['observation_date'] <= end_date)
filtered_data = df_wheat.loc[mask]

# Plot using Plotly
fig_ts = px.line(filtered_data, x='observation_date', y='Wheat_Price', title='Wheat Prices Over Time')
fig_ts.update_layout(xaxis_title='Date', yaxis_title='Wheat Prices')

st.plotly_chart(fig_ts)


# ACF/PACF section
st.subheader("Autocorrelation (ACF) & Partial Autocorrelation (PACF) Analysis")

# Allow users to select number of lags
num_lags = st.slider("Select number of lags:", 1, 50, 20)

# Allow users to select ACF or PACF
plot_type = st.selectbox("Choose plot type:", ["ACF", "PACF"])

# Compute ACF or PACF values based on user selection
if plot_type == "ACF":
    values = acf(df_wheat['Wheat_Price'], nlags=num_lags)
    title = "Autocorrelation Function (ACF)"
else:
    values = pacf(df_wheat['Wheat_Price'], nlags=num_lags)
    title = "Partial Autocorrelation Function (PACF)"

# Plot ACF or PACF using Plotly
lags = list(range(num_lags + 1))  # +1 to include lag 0
fig = go.Figure()
fig.add_trace(go.Scatter(x=lags, y=values, mode='lines+markers'))
fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="Value")
st.plotly_chart(fig)

# # Stationarity Test section
# st.subheader("Stationarity Test")

# # Allow users to select a p-value threshold
# p_value_threshold = st.number_input("Select a p-value threshold:", min_value=0.001, max_value=1.0, value=0.05, step=0.01)

# # Function to check the stationarity of a series based on the chosen p-value
# def check_stationarity(series, threshold):
#     test_results = adfuller(series)
#     if test_results[1] <= threshold:
#         return f"At a p-value of {threshold}, the series is stationary."
#     else:
#         return f"At a p-value of {threshold}, the series is non-stationary."

# # Display the result of the stationarity test
# result = check_stationarity(df_wheat['Wheat_Price'], p_value_threshold)
# st.write(result)



# Modeling and predictions
# Model Forecasting section
st.subheader("Model Forecasting Using a SARIMAX Model")

# Allow users to select training set size
train_size = st.slider("Select size for the training set:", min_value=0.9, max_value=0.99, value=0.9, step=0.01)
split_index = int(train_size * len(df_wheat))
train_data = df_wheat['Wheat_Price'].iloc[:split_index]
test_data = df_wheat['Wheat_Price'].iloc[split_index:]

with st.spinner('Training the SARIMAX model... This might take a while.'):
    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(0, 0, 2, 12)).fit()

# After model fitting, the rest of the code remains the same
forecast = model.forecast(steps=len(test_data))

# Allow users to select colors
forecast_color = st.selectbox("Select a color for the forecasted line:", ["orange", "red", "purple"])
test_data_color = st.selectbox("Select a color for the test set:", ["green", "yellow"])

with st.spinner('Generating plots... This might take a while.'):

    # Plot the data using Plotly
    fig_forecast = go.Figure()

    # Add traces for training data, test data, and forecast
    fig_forecast.add_trace(go.Scatter(x=train_data.index, y=train_data.values, mode='lines', name='Training Data', line=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=test_data.index, y=test_data.values, mode='lines', name='Test Data', line=dict(color=test_data_color)))
    fig_forecast.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='Forecast', line=dict(color=forecast_color)))

    # Set layout and titles
    fig_forecast.update_layout(title='Wheat Prices and Predictions', xaxis_title='Date', yaxis_title='Wheat Price')

    st.plotly_chart(fig_forecast)


# User input for forecast steps
forecast_steps = st.slider("Select number of forecast steps:", 1, 6)

# Forecast using the SARIMAX model beyond the test set
full_model = SARIMAX(df_wheat['Wheat_Price'], order=(1, 1, 1), seasonal_order=(0, 0, 2, 12)).fit()
forecast_values = full_model.forecast(steps=forecast_steps)

# Plot the data using Plotly
fig_forecast_extended = go.Figure()

# Add traces for entire dataset and forecast
fig_forecast_extended.add_trace(go.Scatter(x=df_wheat.index, y=df_wheat['Wheat_Price'].values, mode='lines', name='Actual Data', line=dict(color='blue')))
fig_forecast_extended.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values.values, mode='lines', name='Forecast', line=dict(color='red')))

# Set layout and titles
fig_forecast_extended.update_layout(title='Wheat Prices and Extended Forecast', xaxis_title='Date', yaxis_title='Wheat Price')

st.plotly_chart(fig_forecast_extended)

st.subheader("Model Forecasting Using a Prophet Model")


train_data_prophet = df_wheat.iloc[:split_index]
test_data_prophet = df_wheat.iloc[split_index:]
df_prophet = train_data_prophet
# Prepare dataframe for Prophet
df_prophet.columns = ['ds', 'y']

# Initialize and fit the model
model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
model_prophet.fit(df_prophet)

# Create a dataframe for future dates
forecast_steps_prophet = st.slider("Select number of forecast steps for the Prophet model:", 1, 6)
future = model_prophet.make_future_dataframe(periods=forecast_steps_prophet)

# Forecast
forecast_prophet = model_prophet.predict(future)

# User inputs for colors
in_sample_color = st.selectbox("Select a color for the in-sample forecasted line:", ["orange", "purple", "red"])
out_of_sample_color = st.selectbox("Select a color for the out-of-sample forecasted line:", ["green", "yellow"])

# Plotting with Plotly
fig_prophet = go.Figure()

# Add traces for actual, in-sample forecasted, and out-of-sample forecasted values
fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual', line=dict(color='blue')))
fig_prophet.add_trace(go.Scatter(x=forecast_prophet['ds'][:len(train_data)], y=forecast_prophet['yhat'][:len(train_data)], mode='lines', name='In-Sample Forecast', line=dict(color=in_sample_color)))
fig_prophet.add_trace(go.Scatter(x=forecast_prophet['ds'][len(train_data):], y=forecast_prophet['yhat'][len(train_data):], mode='lines', name='Out-of-Sample Forecast', line=dict(color=out_of_sample_color)))

# # Confidence Intervals
# fig_prophet.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat_upper'], fill='tonexty', mode='lines', fillcolor='rgba(68,68,68,0.2)', line=dict(color='transparent'), name='Upper Confidence Interval'))
# fig_prophet.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat_lower'], fill='tonexty', mode='lines', fillcolor='rgba(68,68,68,0.2)', line=dict(color='transparent'), name='Lower Confidence Interval'))

# Set layout
fig_prophet.update_layout(title='Forecast with Prophet', xaxis_title='Date', yaxis_title='Wheat Prices')

st.plotly_chart(fig_prophet)
