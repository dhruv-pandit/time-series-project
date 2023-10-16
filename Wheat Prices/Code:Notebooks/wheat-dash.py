import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objs as go
import numpy as np
import itertools
import statsmodels.tsa.vector_ar.vecm as vecm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the data
df_wheat = pd.read_excel(r'/Users/dhruvpandit/Documents/GitHub/time-series-project/Wheat Prices/Datasets/PWHEAMTUSDM.xls').rename(columns={'PWHEAMTUSDM' : 'Wheat_Price'})
df_wheat_date = df_wheat.set_index('observation_date')
#df_wheat_train_date, df_wheat_test_date =  df_wheat_date.loc[:int(len(df_wheat_date) * 0.8)], df_wheat_date.loc[int(len(df_wheat_date) * 0.8):]#split dataframe into train and test

df_wheat_train, df_wheat_test =  df_wheat.loc[:int(len(df_wheat) * 0.8)], df_wheat.loc[int(len(df_wheat) * 0.8):]#split dataframe into train and test
#model_auto_2 = SARIMAX(df_wheat_train_date["Wheat_Price"], order=(2, 1, 2), seasonal_order=(1, 1, 2, 12)).fit()
#pred = model_auto_2.predict(start = 314, end = 392)
#forecast_values = model_auto_2.get_forecast(steps=(392-314))
##forecast_ci = forecast_values.conf_int()
#forecast_values = forecast_values.predicted_mean
# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # Header
    html.Header([
        html.H1("Time Series Analysis of Wheat Prices", style={'text-align': 'center', 'font-family' : 'Futura'}),
    ], style={'padding': '20px'}),
    
    # Layout for Variable Selection and Time Series Plots (Left Section)
    html.Div([
        # Variable Selection Section
        # html.Div([
        #     # Variable Dropdown
        #     html.Label("Select Variable(s):"),
        #     dcc.Dropdown(
        #         id="variable-selector",
        #         options=[
        #             {'label': variable, 'value': variable} for variable in data_df.columns
        #         ],
        #         multi=True  # Allow multiple variable selection
        #     ),
        # ]),
        
        # Date Range Picker
        html.Div([
            html.Label("Select Timeframe:"),
            dcc.DatePickerRange(
                id="date-range-selector",
                start_date=df_wheat_date.index.min(),
                end_date=df_wheat_date.index.max()
            ),
        ]),
        
        # Time Series Plots Section
        html.Section([
            dcc.Graph(id="time-series-plots"),
        ]),
    ], style={'width': '45%', 'margin': '10px', 'float': 'left'}),
    
    # Layout for ACF/PACF Selection and Plots (Right Section)
    html.Div([
        # ACF/PACF Selection Section
        html.Div([
            # Radio Buttons for ACF and PACF choice
            html.Label("Select ACF/PACF:"),
            dcc.RadioItems(
                id="acf-pacf-selector",
                options=[
                    {'label': 'ACF', 'value': 'acf'},
                    {'label': 'PACF', 'value': 'pacf'}
                ],
                value='acf',  # Default selection
                labelStyle={'display': 'block'}
            ),

        ]),
        
        # ACF/PACF Plots Section
        html.Section([
            dcc.Graph(id="acf-pacf-plots"),
        ]),
    ], style={'width': '45%', 'margin': '10px', 'float': 'right'}),


], style={'font-family' : 'Futura'})

# Define a callback function to update the Granger test results


# Define callback to update ACF and PACF plots based on user input
@app.callback(
    Output("acf-pacf-plots", "figure"),
    #Input("variable-acf-pacf-selector", "value"),
    Input("acf-pacf-selector", "value")
)
def update_acf_pacf_plots(acf_or_pacf):

    lags = 10  # Number of lags for ACF/PACF
    title = f"{acf_or_pacf.upper()} Plot for Wheat Prices"

    
    
    if acf_or_pacf == 'acf':
        vals_to_plot = sm.tsa.acf(df_wheat['Wheat_Price'], nlags=lags)
        plot_type = 'ACF'
    else:
        vals_to_plot = sm.tsa.pacf(df_wheat['Wheat_Price'], nlags=lags)
        plot_type = 'PACF'

    lags = np.arange(len(vals_to_plot))
    fig = go.Figure(data=[
        go.Bar(x=lags, y=vals_to_plot, name=plot_type)
    ])

    fig.update_layout(title=title, xaxis_title='Lag', yaxis_title=plot_type)

    return fig
# Define callback to update time series plots based on variable selection and timeframe
@app.callback(
    Output("time-series-plots", "figure"),
    Input("date-range-selector", "start_date"),
    Input("date-range-selector", "end_date")
)
def update_time_series_plots(start_date, end_date):

    filtered_df = df_wheat_date.loc[start_date:end_date]
    print(filtered_df)
    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y=filtered_df['Wheat_Price'],
        labels={'index': 'Date', 'value': 'Value'},
        title="Time Series Plot of Wheat Prices"
    )
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
