# ARIMA-GARCH Volatility Forecasting Dashboard

This interactive web application, built with Plotly Dash, provides a comprehensive toolkit for time series analysis and volatility forecasting of major tech stocks. Users can specify a custom ARIMA-GARCH model and evaluate its in-sample forecast performance.

## Methodology Overview

The dashboard guides the user through the standard workflow for building a volatility forecasting model for financial assets.

* **Data Selection & Preprocessing:** The user begins by selecting an equity ticker (from AMZN, AAPL, GOOG, META, MSFT) and a date range for analysis. The application then retrieves the daily close prices and calculates the daily logarithmic returns, which are used for the subsequent modeling steps.

* **Stationarity Analysis:** Log returns are tested for stationarity using the Augmented Dickey-Fuller (ADF) Test. The results, including the test statistic and p-value, are displayed. This step confirms that the series does not have a unit root and is suitable for ARMA modeling. Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots of the stationary series are provided to help identify the appropriate order for the mean model.

* **Mean Model (ARIMA):** Based on the insights from the ACF/PACF plots, the user can specify the parameters p (autoregressive order) and q (moving-average order) for the ARIMA model that will be fitted to the log returns. The order of differencing, d, is determined from the stationarity check.

* **Residual Diagnostics:** After fitting the ARIMA model, its residuals are analyzed to ensure the model has adequately captured the serial correlation in the returns.
    * The **Ljung-Box test** checks for any remaining autocorrelation in the residuals.
    * The **Breusch-Pagan test** checks for heteroskedasticity (i.e., non-constant variance) in the residuals. The presence of heteroskedasticity, also known as ARCH effects, justifies the use of a GARCH model for the volatility.

* **Volatility Model (ARCH/GARCH):** An ARCH(1) and a GARCH(1,1) model are fitted to the residuals of the ARIMA model. This two-step approach (ARIMA -> GARCH) allows for the joint modeling of the conditional mean and conditional variance of the returns. The summary statistics for both volatility models are displayed.

* **In-Sample Forecasting & Evaluation:** The dashboard allows the user to perform an in-sample forecast to evaluate the performance of the fitted GARCH(1,1) model.
    * The user selects a forecast period within the chosen date range.
    * The data is split into a training and testing set. A new ARIMA-GARCH model is fitted using only the training data.
    * A forecast is generated for the test period and plotted against the observed conditional volatility.
    * Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) are calculated to quantify the forecast's accuracy.

## How to Use the App

1.  Navigate to the live app: [https://volatilityforecast.onrender.com/](https://volatilityforecast.onrender.com/)
2.  **Enter a Ticker:** Type one of the available tickers (AMZN, AAPL, GOOG, META, MSFT) and press Enter.
3.  **Select a Date Range:** Use the date pickers to define the start and end dates for the analysis.
4.  **Analyze Stationarity:** Observe the ADF test results and the ACF/PACF plots to determine an appropriate ARIMA specification.
5.  **Specify ARIMA Model:** Input your chosen `p` and `q` values. The dashboard will update with the model's residuals and diagnostic tests.
6.  **Evaluate GARCH Model:** Review the ARCH and GARCH model summaries.
7.  **Assess Forecast Performance:** Use the final set of date pickers to select an in-sample forecast period and view the forecast plot and accuracy metrics.

## Technical Stack

* **Backend & Web Framework:** Dash, Flask
* **Data Manipulation:** Pandas, NumPy
* **Statistical Modeling:**
    * `statsmodels` for ARIMA, ADF test, and diagnostic tests.
    * `arch` for ARCH and GARCH modeling.
    * `scikit-learn` for forecast evaluation metrics.
* **Plotting:** Plotly Express, Plotly Graph Objects
* **Deployment:** Render
