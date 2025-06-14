import pandas as pd
import numpy as np
import os
import pickle
from dash import Dash, html, dcc, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Loads data
pickle_filename = "stock_analysis_data.pkl"
file = open(pickle_filename, 'rb')
loaded_data = pickle.load(file)
file.close()
df = pd.DataFrame.from_dict(loaded_data["stock_data"])
df.index = pd.to_datetime(loaded_data["times"])

initial_ticker = "AMZN" # Sets default ticker

app = Dash(__name__)
server = app.server

colors = {
    'background': "#FFFFFF",
    'text': "#000000"
}

# --- App Layout ---
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    # Ticker Input
    html.H3("Type ticker (AMZN, AAPL, GOOG, META, MSFT) and press Enter:",
        style={'textAlign': 'center'}),
    html.Div(style={'textAlign': 'center'}, children=[
        dcc.Input(id='ticker-input', debounce=True, value=initial_ticker, type='text',
            style={'textAlign': 'center'})
]),

    # Date Picker Inputs
    html.Div([
        html.Div([
            html.H6("Start Date:"),
            dcc.DatePickerSingle(
                id='start-date-picker',
                min_date_allowed=df.index.min().date(),
                max_date_allowed=df.index.max().date(),
                initial_visible_month=df.index.min().date(),
                date=df.index.min().date()
            ),
        ], style={'display': 'inline-block'}),
        
        html.Div([
            html.H6("End Date:"),
            dcc.DatePickerSingle(
                id='end-date-picker',
                min_date_allowed=df.index.min().date(),
                max_date_allowed=df.index.max().date(),
                initial_visible_month=df.index.max().date(),
                date=df.index.max().date()
            ),
        ], style={'display': 'inline-block'}),
    ], style={'textAlign': 'center'}),

    html.H1(
        id='dash-title',
        style={'textAlign': 'center'}
    ),

    html.Div(
        id='description-text',
        style={'textAlign': 'center'}
    ),

    dcc.Graph(id='price-chart'),
    dcc.Graph(id='return-chart'),

    #ADF test
    html.H3("Augmented Dickey-Fuller Test for Stationarity",
        style={'textAlign': 'center'}),
    html.Div(
        id='ADF_results',
        style={'textAlign': 'center'}
    ),

    html.Div([
        dcc.Graph(id='PACF',
                  style={'display': 'inline-block', 'width':'49%'}),
        dcc.Graph(id='ACF',
                  style={'display': 'inline-block', 'width':'49%'})
    ], style={'textAlign': 'center'}),

    # ARIMA Specification Inputs
    html.Div([
        html.H3("Specify ARIMA parameters and press Enter (d already calculated):",
            style={'textAlign': 'center'}),

        html.Div([
            html.H6("p (AR component):"),
            dcc.Input(
                id='p_arima', debounce=True, value=0, type='number'
            ),
        ], style={'display': 'inline-block'}),
        
        html.Div([
            html.H6("q (MA component):"),
            dcc.Input(
                id='q_arima', debounce=True, value=0, type='number'
            ),
        ], style={'display': 'inline-block'}),
        
        html.Div(
        id='arima_text',
        style={'textAlign': 'center', 'padding-top': '10px'}
        )
    ], style={'textAlign': 'center'}),

    #. Residual plots
    html.Div([
        dcc.Graph(id='residual_plot',
                  style={'display': 'inline-block', 'width':'49%'}),
        dcc.Graph(id='residual_sq_plot',
                  style={'display': 'inline-block','width':'49%'})
    ], style={'textAlign': 'center'}),

    # Ljung-Box + Breusch-Pagan tests
    html.H3("Ljung-Box Test for Serial Correlation of Residuals",
        style={'textAlign': 'center'}),
    html.Div(
        id='lb_result',
        style={'textAlign': 'center'}),
    html.H3("Breusch-Pagan Test For Heteroskedasticity of Residuals",
        style={'textAlign': 'center'}),
    html.Div(
        id='bp_result',
        style={'textAlign': 'center'}),

    # ARCH model
    html.H3("ARCH(1) model",
        style={'textAlign': 'center'}),
    html.Div(children=[
        html.Pre(id='ARCH_result')
        ],
        style={'textAlign': 'center'}),

    # GARCH model
    html.H3("GARCH(1,1) model",
        style={'textAlign': 'center'}),
    html.Div(children=[
        html.Pre(id='GARCH_result')
        ],
        style={'textAlign': 'center'}),
    
    # Forecast Date Picker and Metrics
    html.Div([
        html.H2("GARCH Forecast Performance",
                style={'textAlign': 'center'}),
        html.H4("Select a date range for the in-sample forecast:", 
                style={'textAlign': 'center'}),
        html.Div([
            dcc.DatePickerSingle(
                id='forecast-start-date-picker',
                min_date_allowed=df.index.min().date(),
                max_date_allowed=df.index.max().date(),
                date=df.index.max().date() - pd.Timedelta(days=30), # Default start
                style={'marginRight': '10px'}
            ),
            dcc.DatePickerSingle(
                id='forecast-end-date-picker',
                min_date_allowed=df.index.min().date(),
                max_date_allowed=df.index.max().date(),
                date=df.index.max().date(), # Default end
            ),
        ], style={'display': 'inline-block'}),

    ], style={'textAlign': 'center'}),

    dcc.Graph(id='vol_forecast_plot'),

    html.Div(
            id='forecast-metrics',
            style={'textAlign': 'center', 'fontWeight': 'bold', 'padding-bottom': '20px'}
        ),
])

# --- Callback to Update the Graphs and Text ---
@callback(
    Output('price-chart', 'figure'),
    Output('return-chart', 'figure'),
    Output('dash-title', 'children'),
    Output('description-text', 'children'),
    Output('ADF_results', 'children'),
    Output('ACF', 'figure'),
    Output('PACF', 'figure'),
    Output('arima_text', 'children'),
    Output('residual_plot', 'figure'),
    Output('residual_sq_plot', 'figure'),
    Output('lb_result', 'children'),
    Output('bp_result', 'children'),
    Output('ARCH_result', 'children'),
    Output('GARCH_result', 'children'),
    Output('forecast-metrics', 'children'),
    Output('vol_forecast_plot', 'figure'),
    Input('ticker-input', 'value'),
    Input('start-date-picker', 'date'),
    Input('end-date-picker', 'date'),
    Input('forecast-start-date-picker', 'date'),
    Input('forecast-end-date-picker', 'date'),
    Input('p_arima', 'value'),
    Input('q_arima', 'value'),
)

def update_graphs(input_ticker, start_date, end_date, forecast_start_date, forecast_end_date, p, q):
    """
    Updates the dashboard components based on the user's ticker input.
    """
    # Use the input value, fall back to the initial ticker if input is empty
    ticker = str(input_ticker).upper().strip() if input_ticker else initial_ticker
    
    # Check if the requested ticker is in our data
    if ticker not in df.columns:
        # If not, return an empty chart and an error message
        error_title = f"Ticker '{ticker}' not found"
        error_desc = "Please enter a valid ticker from the dataset."
        # Return empty outputs for all 16 outputs
        return {}, {}, error_title, error_desc, "", {}, {}, "", {}, {}, "", "", "", "", "", {}
    
    dff = df.loc[start_date:end_date]
    log_returns = (np.log(dff[ticker]).diff() * 100).dropna() # type: ignore
#################################
    # --- Price and Return Charts ---
    # Price chart
    price_chart = px.line(
        x=dff.index,
        y=dff[ticker],
        title=f"{ticker} Daily Close Price"
    )
    price_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    price_chart.update_xaxes(rangeslider_visible=True)

    # Daily logarithmic returns chart
    return_chart = px.line(
        x=log_returns.index,
        y=log_returns,
        title=f"{ticker} Daily Log Returns"
    )
    return_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Log Return")
    return_chart.update_xaxes(rangeslider_visible=True)
    # --- End of Price and Return Charts ---
#################################
    # --- ADF test for stationarity ---
    def get_adf_result(series):
        return adfuller(series.dropna(), autolag='AIC')
        
    adf_iterations_log = []
    
    # Test initial series
    adf_result = get_adf_result(log_returns)
    p_value = adf_result[1]
    is_stationary = p_value <= 0.05
    conclusion = "The series is likely stationary." if is_stationary else "The series is likely non-stationary."
    adf_iterations_log.append(
        f"Test on Log Returns (d = 0):\n"
        f"ADF Statistic: {adf_result[0]:.4f} | p-value: {p_value:.4f}\n"
        f"Conclusion: {conclusion}"
    )
    
    prices_diff = log_returns.copy()
    d = 0 # Order of differencing

    # Loop to difference the series until it is stationary
    while not is_stationary:
        d += 1
        prices_diff = prices_diff.diff().dropna()
        if len(prices_diff) < 2:
            adf_iterations_log.append("\nNot enough data to continue differencing.")
            break
            
        adf_result = get_adf_result(prices_diff)
        p_value = adf_result[1]
        is_stationary = p_value <= 0.05
        conclusion = "The series is likely stationary." if is_stationary else "The series is likely non-stationary."
        adf_iterations_log.append(
            f"\n---\nTest on Differenced Series (d = {d}):\n"
            f"ADF Statistic: {adf_result[0]:.4f} | p-value: {p_value:.4f}\n"
            f"Conclusion: {conclusion}"
        )
        
        if d > 2: # Avoid over-differencing
            if not is_stationary:
                 adf_iterations_log.append("\nStopping after 3 differences to prevent over-differencing.")
            break
    
    # Combine all ADF iteration logs into a single string
    ADF_results = "\n".join(adf_iterations_log)
    # --- End of ADF test ---
#################################
    # --- ACF and PACF Plots ---
    def create_corr_plot(series, plot_pacf=False):
        # Ensure series is not empty before proceeding
        if series.empty:
            fig = go.Figure()
            title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
            fig.update_layout(title=title, annotations=[dict(text="Not enough data to display plot", showarrow=False)])
            return fig
            
        corr_array = pacf(series.dropna(), alpha=0.05) if plot_pacf else acf(series.dropna(), alpha=0.05)
        lower_y = corr_array[1][:,0] - corr_array[0]
        upper_y = corr_array[1][:,1] - corr_array[0]

        fig = go.Figure()
        [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
        for x in range(len(corr_array[0]))]
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                    marker_size=12)
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
                fill='tonexty', line_color='rgba(255,255,255,0)')
        fig.update_traces(showlegend=False)
        fig.update_xaxes(range=[-1,27])
        fig.update_yaxes(zerolinecolor='#000000')
        
        title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
        fig.update_layout(title=title)
        return fig
    
    pacf_plot = create_corr_plot(prices_diff, plot_pacf=True)
    acf_plot = create_corr_plot(prices_diff, plot_pacf=False)
    # --- End of ACF and PACF Plots ---
#################################
    # --- ARIMA Specification and Residual Diagnostics
    model = ARIMA(log_returns, order=(p, d, q))
    model_fit = model.fit()
    arima_text = (
        f'The ARIMA model specified is ARIMA({p}, {d}, {q}).'
    )

    # Residual plots. First d residuals are dropped due to lack of data if series is integrated
    residuals = model_fit.resid[d:]
    residuals_plot = px.scatter(
        x=residuals.index,
        y=residuals,
        title=f"ARIMA Model Residuals"
    )
    residuals_plot.update_xaxes(rangeslider_visible=True)
    residuals_plot.update_layout(
        xaxis_title="Date",
        yaxis_title="Residual"
    )
    residuals_sq_plot = px.scatter(
        x=residuals.index,
        y=residuals ** 2,
        title=f"ARIMA Model Squared Residuals"
    )
    residuals_sq_plot.update_layout(
        xaxis_title="Date",
        yaxis_title="Squared Residual"
    )
    residuals_sq_plot.update_xaxes(rangeslider_visible=True)

    # Ljung-Box test on residuals for serial correlation
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True) # Check up to 10 lags
    if lb_test['lb_pvalue'].iloc[0] > 0.05:
        lb_result = f"""Ljung-Box test statistic: {lb_test['lb_stat'].iloc[0]:.4}; p = {lb_test['lb_pvalue'].iloc[0]:.4}; residuals appear to be white noise without serial correlation."""
    else:
        lb_result = f"""Ljung-Box test statistic: {lb_test['lb_stat'].iloc[0]:.4}; p = {lb_test['lb_pvalue'].iloc[0]:.4}; residuals may not be white noise and show serial correlation."""

    # Breusch-Pagan test for heteroskedasticity of residuals
    exog_het_time = pd.DataFrame({'const': np.ones(len(residuals)), 'time_trend': np.arange(len(residuals))})
    bp_test = het_breuschpagan(residuals, exog_het_time)
    if bp_test[1] > 0.05:
        bp_result = f"LM Statistic: {bp_test[0]:.4}; p = {bp_test[1]:.4}; residuals appear to be homoskedastic. GARCH volatility forecasting may not be appropriate."
    else:
        bp_result = f"LM Statistic: {bp_test[0]:.4}; p = {bp_test[1]:.4}; residuals may be be heteroskedastic. GARCH volatility forecasting may be appropriate."

    # --- End of ARIMA Specification and Residual Diagnostics
#################################
    # --- ARCH Test, GARCH Fitting
    arch_spec = arch_model(log_returns, mean='Zero', vol='ARCH', p=1)
    arch_results = arch_spec.fit(disp='off')
    ARCH_result = arch_results.summary().as_text()

    garch_spec = arch_model(log_returns, mean='Zero', vol='GARCH', p=1, q=1, dist='t')
    garch_results = garch_spec.fit(disp='off')
    GARCH_result = garch_results.summary().as_text()

    # --- End of ARCH Test, GARCH Fitting
#################################
    # --- GARCH Volatility Forecasting ---
    
    # 1. Get Observed Volatility from the full model fitted on the entire selected date range
    observed_vol = garch_results.conditional_volatility

    # 2. Split data for training and testing based on the forecast dates
    train_returns = log_returns.loc[:forecast_start_date]
    test_index = log_returns.loc[forecast_start_date:forecast_end_date].index
    
    # 3. Fit a new GARCH model on the training data only
    garch_train_spec = arch_model(train_returns, mean='Zero', vol='GARCH', p=1, q=1, dist='t')
    garch_train_results = garch_train_spec.fit(disp='off')
    
    # 4. Forecast from the end of the training data and calculate metrics
    horizon = len(test_index)
    forecasted_volatility = pd.Series(dtype='float64') # Initialize empty series
    metrics_text = "Select a forecast period to see performance metrics."

    if horizon > 0:
        forecast = garch_train_results.forecast(horizon=horizon, reindex=False)
        forecasted_volatility = np.sqrt(forecast.variance.iloc[-1])
        forecasted_volatility.index = test_index # type: ignore
        
        # Calculate metrics
        observed = observed_vol.loc[test_index] # type: ignore
        forecasted = forecasted_volatility
        mse = mean_squared_error(observed, forecasted)
        mape = mean_absolute_percentage_error(observed, forecasted)
        metrics_text = f"Forecast MSE: {mse:.4f}  |  Forecast MAPE: {mape:.2%}"

    # Create plot
    vol_forecast_plot = go.Figure()

    # Add historical/training conditional volatility
    vol_forecast_plot.add_trace(go.Scatter(
        x=train_returns.index,
        y=observed_vol.loc[train_returns.index], # type: ignore
        mode='lines',
        name='Historical (Training) Volatility',
        line=dict(color='royalblue', width=2)
    ))
    
    # Only add test & forecast traces if there is a forecast period
    if horizon > 0:
        # Add observed volatility during the forecast period
        vol_forecast_plot.add_trace(go.Scatter(
            x=test_index,
            y=observed_vol.loc[test_index], # type: ignore
            mode='lines',
            name='Observed (In-Sample) Volatility',
            line=dict(color='mediumseagreen', width=2)
        ))

        # Add the forecasted volatility
        vol_forecast_plot.add_trace(go.Scatter(
            x=forecasted_volatility.index, # type: ignore
            y=forecasted_volatility,
            mode='lines',
            name='Forecasted Volatility',
            line=dict(color='darkorange', dash='dash', width=2)
        ))

    # Update layout
    vol_forecast_plot.update_layout(
        title={
            'text': 'GARCH(1,1) In-Sample Volatility Forecast vs. Observed',
            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='Conditional Volatility (%)',
        legend_title='Volatility Type'
    )
    vol_forecast_plot.update_xaxes(rangeslider_visible=True)

    # --- End of GARCH Volatility Forecasting ---
#################################

    # Strings for start/end date
    start_date_str = start_date
    end_date_str = end_date
    
    dash_title = (f"Analysis for: {ticker}")
    description_text = (
        f"This dashboard shows the daily price and returns for {ticker} from {start_date_str} to {end_date_str}, "
        "then conducts a forecast of volatility.\n"
        "The order of differencing to make the price time series stationary is determined using Augmented Dickey-Fuller test, "
        "then ACF and PACF plots are shown to determine specification of ARIMA model.\n"
        "Residual plots are shown for model diagnostics, with Ljung-Box and Bruesch-Pagan tests performed."
        "ARCH effects are tested for, then a GARCH model is fitting for volatility forecasting."
        "GARCH forecasts of conditional volatility are shown, then MSE and MAPE are displayed for the specified forecasts."
    )
    
    return price_chart, return_chart, dash_title, description_text, ADF_results, pacf_plot, acf_plot, arima_text, residuals_plot, residuals_sq_plot, lb_result, bp_result, ARCH_result, GARCH_result, metrics_text, vol_forecast_plot

if __name__ == '__main__':
    app.run(debug=True)