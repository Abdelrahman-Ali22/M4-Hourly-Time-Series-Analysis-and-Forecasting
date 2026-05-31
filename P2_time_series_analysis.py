"""
Business Intelligence Workplace Project 2
Time Series Analysis and Forecasting for M4 Hourly Series H83

Required input files in the same folder:
- Hourly-train.csv
- Hourly-test.csv

Outputs saved by the script:
- H83_forecast_results.csv
- H83_accuracy_comparison.csv
- H83_ARIMA_model_selection.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


# ==============================
# SETTINGS
# ==============================

SERIES_NAME = "H83"
PERIOD = 24
POLY_DEGREE = 3
FORECAST_STEPS = 24

TRAIN_FILE = Path("Hourly-train.csv")
TEST_FILE = Path("Hourly-test.csv")


def check_required_files():
    """Check that the required M4 dataset files exist."""
    missing_files = []

    for file_path in [TRAIN_FILE, TEST_FILE]:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(
            "Missing required dataset file(s): "
            + ", ".join(missing_files)
            + "\nPlease put Hourly-train.csv and Hourly-test.csv in the same folder as this script."
        )


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    check_required_files()

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    train_row = train_data[train_data["V1"] == SERIES_NAME]
    test_row = test_data[test_data["V1"] == SERIES_NAME]

    if train_row.empty:
        raise ValueError(f"The selected series {SERIES_NAME} was not found in the training dataset.")

    if test_row.empty:
        raise ValueError(f"The selected series {SERIES_NAME} was not found in the test dataset.")

    # Full training set for descriptive analysis, decomposition, residual analysis, and ARIMA modeling
    x = train_row.iloc[0, 1:].dropna().astype(float).values

    # First 24 observations from the test set for predictive analysis
    real_test = test_row.iloc[0, 1:].dropna().astype(float).values[:FORECAST_STEPS]

    print("Series used:", SERIES_NAME)
    print("Number of training observations:", len(x))
    print("Number of test observations used:", len(real_test))

    # ==============================
    # DESCRIPTIVE ANALYSIS
    # ==============================

    print("\n==============================")
    print("DESCRIPTIVE ANALYSIS")

    print("Mean:", np.mean(x))
    print("Median:", np.median(x))
    print("Standard deviation:", np.std(x, ddof=1))
    print("Minimum:", np.min(x))
    print("Maximum:", np.max(x))
    print("First quartile Q1:", np.percentile(x, 25))
    print("Third quartile Q3:", np.percentile(x, 75))
    print("Interquartile range:", np.percentile(x, 75) - np.percentile(x, 25))

    plt.figure(figsize=(12, 4))
    plt.plot(x)
    plt.title(f"{SERIES_NAME} Hourly Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(x, bins=30)
    plt.title(f"Histogram of {SERIES_NAME}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.boxplot(x)
    plt.title(f"Boxplot of {SERIES_NAME}")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

    plot_acf(x, lags=48)
    plt.title("ACF Plot of Original Series")
    plt.grid(True)
    plt.show()

    plot_pacf(x, lags=48, method="ywm")
    plt.title("PACF Plot of Original Series")
    plt.grid(True)
    plt.show()

    # ==============================
    # CLASSICAL DECOMPOSITION
    # Xt = mt + st + Yt
    # ==============================

    print("\n==============================")
    print("CLASSICAL DECOMPOSITION")

    # Since the data are hourly, d = 24 represents one daily seasonal cycle.
    period = PERIOD
    n = len(x)

    # For even period d = 24, d = 2q, so q = 12.
    half_period = period // 2

    # Trend estimation using weighted moving average.
    m_hat = np.empty(n)
    m_hat[:] = np.nan

    for t in range(half_period, n - half_period):
        window = x[t - half_period : t + half_period + 1]
        weighted_sum = 0.5 * window[0] + np.sum(window[1:-1]) + 0.5 * window[-1]
        m_hat[t] = weighted_sum / period

    plt.figure(figsize=(12, 4))
    plt.plot(x, label="Original data")
    plt.plot(m_hat, label="Moving average trend", linewidth=2)
    plt.title("Original Data and Moving Average Trend")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Seasonal component estimation.
    w = []

    for k in range(period):
        deviations = []

        for i in range(k, n, period):
            if not np.isnan(m_hat[i]):
                deviations.append(x[i] - m_hat[i])

        w.append(np.mean(deviations))

    w = np.array(w)

    # Adjust seasonal component so that the 24 seasonal values sum to zero.
    seasonal_pattern = w - np.mean(w)

    print("\nSeasonal pattern for 24 hours:")
    print(seasonal_pattern)
    print("Sum of seasonal pattern:", np.sum(seasonal_pattern))

    s_hat = np.empty(n)

    for i in range(n):
        s_hat[i] = seasonal_pattern[i % period]

    plt.figure(figsize=(12, 4))
    plt.plot(seasonal_pattern, marker="o")
    plt.title("Estimated 24 Hour Seasonal Pattern")
    plt.xlabel("Hour of Daily Cycle")
    plt.ylabel("Seasonal Value")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(s_hat)
    plt.title("Estimated Seasonal Component for Full Series")
    plt.xlabel("Time")
    plt.ylabel("Seasonal Value")
    plt.grid(True)
    plt.show()

    # Trend reestimation using polynomial fit.
    d_t = x - s_hat
    time_axis = np.arange(n)

    poly_coeffs = np.polyfit(time_axis, d_t, deg=POLY_DEGREE)
    m_refined = np.polyval(poly_coeffs, time_axis)

    plt.figure(figsize=(12, 4))
    plt.plot(d_t, label="Deseasonalized data")
    plt.plot(m_refined, label="Polynomial trend", linewidth=2)
    plt.title("Deseasonalized Data and Polynomial Trend")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Noise / residual estimation:
    # Y_hat_t = X_t - m_hat_t - s_hat_t
    # Since d_t = X_t - s_hat_t, then Y_hat_t = d_t - m_refined.
    residuals = d_t - m_refined

    plt.figure(figsize=(12, 4))
    plt.plot(residuals)
    plt.title("Residual Series after Classical Decomposition")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    sm.qqplot(residuals, line="s")
    plt.title("QQ Plot of Residuals")
    plt.grid(True)
    plt.show()

    plot_acf(residuals, lags=48)
    plt.title("ACF Plot of Residuals")
    plt.grid(True)
    plt.show()

    plot_pacf(residuals, lags=48, method="ywm")
    plt.title("PACF Plot of Residuals")
    plt.grid(True)
    plt.show()

    # ==============================
    # RESIDUAL ANALYSIS
    # ==============================

    print("\n==============================")
    print("RESIDUAL ANALYSIS")

    res_mean = np.mean(residuals)
    res_std = np.std(residuals, ddof=1)

    print("Residual mean:", res_mean)
    print("Residual standard deviation:", res_std)

    # ADF test for stationarity.
    # H0: the series has a unit root and is not stationary.
    adf_result = adfuller(residuals)

    print("\nADF TEST FOR STATIONARITY")
    print("ADF statistic:", adf_result[0])
    print("ADF p-value:", adf_result[1])

    if adf_result[1] < 0.05:
        print("Residuals are stationary.")
    else:
        print("Residuals are not stationary.")

    # Ljung-Box test for independence.
    # H0: no autocorrelation up to the selected lag.
    ljung_result = acorr_ljungbox(residuals, lags=[24, 48], return_df=True)

    print("\nLJUNG-BOX TEST FOR INDEPENDENCE")
    print(ljung_result)

    if ljung_result["lb_pvalue"].iloc[0] < 0.05:
        print("Residuals are not independent.")
        print("This means some time dependence remains, so ARIMA modeling is useful.")
    else:
        print("Residuals look independent at lag 24.")
        print("This means most time structure was already removed.")

    # Jarque-Bera test for normality.
    # H0: residuals are normally distributed.
    jb_stat, jb_pvalue = jarque_bera(residuals)

    print("\nJARQUE-BERA TEST FOR NORMALITY")
    print("JB statistic:", jb_stat)
    print("JB p-value:", jb_pvalue)

    if jb_pvalue < 0.05:
        print("Residuals are not normally distributed.")
    else:
        print("Residuals look normally distributed.")

    # ==============================
    # ARIMA MODEL SELECTION
    # ==============================

    print("\n==============================")
    print("ARIMA MODEL SELECTION")

    best_aic = np.inf
    best_bic = np.inf
    best_order_aic = None
    best_order_bic = None
    best_model_aic = None
    best_model_bic = None

    model_results = []

    for p in range(0, 5):
        for q in range(0, 5):
            try:
                arima_model = ARIMA(residuals, order=(p, 0, q))
                arima_result = arima_model.fit()

                model_results.append([p, 0, q, arima_result.aic, arima_result.bic])

                print("ARIMA", (p, 0, q), "AIC =", arima_result.aic, "BIC =", arima_result.bic)

                if arima_result.aic < best_aic:
                    best_aic = arima_result.aic
                    best_order_aic = (p, 0, q)
                    best_model_aic = arima_result

                if arima_result.bic < best_bic:
                    best_bic = arima_result.bic
                    best_order_bic = (p, 0, q)
                    best_model_bic = arima_result

            except Exception as e:
                print("ARIMA", (p, 0, q), "did not work.")
                print("Reason:", e)

    model_selection_table = pd.DataFrame(
        model_results,
        columns=["p", "d", "q", "AIC", "BIC"]
    )

    print("\nARIMA MODEL SELECTION TABLE SORTED BY AIC")
    print(model_selection_table.sort_values("AIC"))

    print("\nBEST MODEL ACCORDING TO AIC")
    print("Best order:", best_order_aic)
    print("Best AIC:", best_aic)

    print("\nBEST MODEL ACCORDING TO BIC")
    print("Best order:", best_order_bic)
    print("Best BIC:", best_bic)

    # Final model selected according to the lowest AIC.
    best_model = best_model_aic
    best_order = best_order_aic

    print("\nFINAL SELECTED ARIMA MODEL")
    print("Selected order:", best_order)
    print(best_model.summary())

    # ==============================
    # DIAGNOSTICS OF SELECTED ARIMA MODEL
    # ==============================

    print("\n==============================")
    print("DIAGNOSTICS OF SELECTED ARIMA MODEL")

    arima_residuals = best_model.resid

    plt.figure(figsize=(12, 4))
    plt.plot(arima_residuals)
    plt.title("Residuals of the Selected ARIMA Model")
    plt.xlabel("Time")
    plt.ylabel("ARIMA Residual")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(arima_residuals, bins=30)
    plt.title("Histogram of ARIMA Residuals")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    sm.qqplot(arima_residuals, line="s")
    plt.title("QQ Plot of ARIMA Residuals")
    plt.grid(True)
    plt.show()

    plot_acf(arima_residuals, lags=48)
    plt.title("ACF Plot of ARIMA Residuals")
    plt.grid(True)
    plt.show()

    arima_ljung = acorr_ljungbox(arima_residuals, lags=[24, 48], return_df=True)

    print("\nLJUNG-BOX TEST FOR ARIMA RESIDUALS")
    print(arima_ljung)

    if arima_ljung["lb_pvalue"].iloc[0] < 0.05:
        print("Some autocorrelation remains in the ARIMA residuals.")
    else:
        print("ARIMA residuals look independent at lag 24.")

    # ==============================
    # PREDICTIVE ANALYSIS
    # ==============================

    print("\n==============================")
    print("PREDICTIVE ANALYSIS")

    # Forecast residuals using selected ARIMA model.
    res_forecast = best_model.forecast(steps=FORECAST_STEPS)

    # Forecast trend and seasonality.
    future_time = np.arange(n, n + FORECAST_STEPS)
    future_trend = np.polyval(poly_coeffs, future_time)

    future_seasonality = np.empty(FORECAST_STEPS)

    for i in range(FORECAST_STEPS):
        future_seasonality[i] = seasonal_pattern[(n + i) % period]

    # Final combined forecast:
    # Forecast = trend forecast + seasonal forecast + residual forecast.
    final_forecast = future_trend + future_seasonality + res_forecast

    # Simple forecasting methods.
    mean_forecast = np.repeat(np.mean(x), FORECAST_STEPS)
    naive_forecast = np.repeat(x[-1], FORECAST_STEPS)
    seasonal_naive_forecast = x[-FORECAST_STEPS:]

    # Evaluation metrics.
    accuracy_table = pd.DataFrame({
        "Method": [
            "Decomposition + ARIMA",
            "Mean forecast",
            "Naive forecast",
            "Seasonal naive forecast"
        ],
        "MAE": [
            mae(real_test, final_forecast),
            mae(real_test, mean_forecast),
            mae(real_test, naive_forecast),
            mae(real_test, seasonal_naive_forecast)
        ],
        "RMSE": [
            rmse(real_test, final_forecast),
            rmse(real_test, mean_forecast),
            rmse(real_test, naive_forecast),
            rmse(real_test, seasonal_naive_forecast)
        ],
        "MAPE (%)": [
            mape(real_test, final_forecast),
            mape(real_test, mean_forecast),
            mape(real_test, naive_forecast),
            mape(real_test, seasonal_naive_forecast)
        ]
    })

    print("\nACCURACY COMPARISON")
    print(accuracy_table)

    # Forecast comparison plot.
    hours = np.arange(1, FORECAST_STEPS + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(hours, real_test, marker="o", label="Actual test data", linewidth=2)
    plt.plot(hours, final_forecast, marker="o", label="Decomposition + ARIMA")
    plt.plot(hours, mean_forecast, marker="o", label="Mean forecast")
    plt.plot(hours, naive_forecast, marker="o", label="Naive forecast")
    plt.plot(hours, seasonal_naive_forecast, marker="o", label="Seasonal naive forecast")
    plt.title(f"24 Hour Forecast Comparison for {SERIES_NAME}")
    plt.xlabel("Forecast Hour")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    results_table = pd.DataFrame({
        "Hour": hours,
        "Actual_Test_Value": real_test,
        "Decomposition_ARIMA_Forecast": final_forecast,
        "Mean_Forecast": mean_forecast,
        "Naive_Forecast": naive_forecast,
        "Seasonal_Naive_Forecast": seasonal_naive_forecast
    })

    print("\nFORECAST TABLE")
    print(results_table)

    forecast_file = f"{SERIES_NAME}_forecast_results.csv"
    accuracy_file = f"{SERIES_NAME}_accuracy_comparison.csv"
    model_selection_file = f"{SERIES_NAME}_ARIMA_model_selection.csv"

    results_table.to_csv(forecast_file, index=False)
    accuracy_table.to_csv(accuracy_file, index=False)
    model_selection_table.to_csv(model_selection_file, index=False)

    print("\nSaved files:")
    print(forecast_file)
    print(accuracy_file)
    print(model_selection_file)


if __name__ == "__main__":
    main()
