# M4 Hourly Time Series Analysis and Forecasting

## Project Overview

This project performs a complete time series analysis using an hourly series from the **M4 Forecasting Competition Dataset**. The project includes descriptive analysis, classical decomposition, residual analysis, ARIMA model selection, forecasting, and comparison with simple benchmark forecasting methods.

The project was completed as part of **Business Intelligence Workplace Project 2**.

## Assignment Objective

The aim of this project is to conduct a comprehensive time series analysis, including statistical modeling and forecasting.

The required tasks include:

* Descriptive analysis of the selected hourly time series
* Classical decomposition into trend, seasonal, and noise components
* Residual analysis
* ARIMA model selection for the residual series
* Forecasting the next 24 observations
* Comparing ARIMA forecasts with simple forecasting methods:

  * Mean forecast
  * Naive forecast
  * Seasonal naive forecast

## Dataset

The dataset used in this project is the **M4 Forecasting Competition Dataset**.

The project uses the hourly dataset files:

```bash
Hourly-train.csv
Hourly-test.csv
```

The training data is used for descriptive analysis, decomposition, residual analysis, and model fitting. The first 24 observations of the test set are used to evaluate the forecasting performance.

## Technologies Used

The project was implemented in Python using the following libraries:

```python
pandas
numpy
matplotlib
scipy
statsmodels
scikit-learn
```

## Project Workflow

### 1. Data Loading

The train and test datasets are loaded using `pandas`.

```python
train_df = pd.read_csv("Hourly-train.csv")
test_df = pd.read_csv("Hourly-test.csv")
```

The selected time series is extracted from the hourly dataset and converted into a numeric time series format.

### 2. Descriptive Analysis

The project begins with a descriptive overview of the selected time series.

This includes:

* Plotting the full training time series
* Observing the general movement of the data
* Checking for visible trend
* Checking for seasonal patterns
* Identifying unusual changes or fluctuations

The goal of this step is to understand the main behavior of the time series before modeling.

### 3. Classical Decomposition

The time series is decomposed into three main components:

```text
Xt = mt + st + Yt
```

Where:

* `mt` is the trend component
* `st` is the seasonal component
* `Yt` is the noise or residual component

The decomposition helps separate the long-term movement, repeated seasonal behavior, and random variation in the data.

### 4. Trend Estimation Using Moving Average

The first trend estimate is calculated using a moving average method.

For an even seasonal period, the centered moving average formula is used:

```text
m̂t = (0.5Xt-q + Xt-q+1 + ... + Xt+q-1 + 0.5Xt+q) / d
```

This smooths the time series and helps estimate the slowly changing trend component.

### 5. Seasonal Component Estimation

After estimating the trend, the seasonal component is calculated by measuring the average deviation from the trend for each seasonal position.

The seasonal component is adjusted so that the seasonal values sum to zero over one full period.

This step helps estimate the repeating seasonal pattern in the time series.

### 6. Polynomial Trend Reestimation

After removing the seasonal component, the deseasonalized data is calculated as:

```text
dt = Xt - ŝt
```

A polynomial trend is then fitted to the deseasonalized data.

This gives a smoother and more flexible estimate of the trend compared with the moving average.

### 7. Noise Estimation

The estimated noise series is calculated as:

```text
Ŷt = Xt - m̂t - ŝt
```

This residual series represents the remaining part of the time series after removing the trend and seasonal components.

### 8. Residual Analysis

The residuals are analyzed to check whether they behave like independent and identically distributed random variables.

The analysis includes:

* Residual time plot
* Histogram of residuals
* QQ-plot
* Autocorrelation function, ACF
* Statistical checks

This step is important because if the residuals still contain structure or autocorrelation, then additional modeling is needed.

### 9. ARIMA Model Selection

An ARIMA model is fitted to the residual series.

The goal is to model the remaining dependence in the residuals after decomposition.

Different ARIMA models are compared, and the best model is selected based on model performance and diagnostic results.

### 10. Predictive Analysis

The selected model is used to forecast the next 24 observations.

The forecast is then compared with the first 24 values from the test dataset.

The model performance is evaluated using forecasting error metrics such as:

* Mean Absolute Error, MAE
* Mean Squared Error, MSE
* Root Mean Squared Error, RMSE

### 11. Benchmark Forecasting Methods

The ARIMA forecast is compared with simple forecasting methods.

#### Mean Forecast

The mean forecast uses the average value of the training series as the prediction.

#### Naive Forecast

The naive forecast uses the last observed value from the training series as the prediction for future values.

#### Seasonal Naive Forecast

The seasonal naive forecast uses the value from the same seasonal position in the previous period.

These methods are useful as benchmarks because a more advanced model should perform better than simple forecasting methods.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Open the Project Folder

```bash
cd your-repository-name
```

### 3. Install the Required Libraries

```bash
pip install pandas numpy matplotlib scipy statsmodels scikit-learn
```

### 4. Add the Dataset Files

Make sure these files are in the same folder as the code:

```bash
Hourly-train.csv
Hourly-test.csv
```

### 5. Run the Python Script

```bash
python P2_time_series_analysis.py
```

Or open the notebook:

```bash
jupyter notebook P2bi_clean.ipynb
```

Then run the notebook cells from top to bottom.

## Repository Structure

```bash
.
├── README.md
├── P2_time_series_analysis.py
├── P2bi_clean.ipynb
├── Hourly-train.csv
└── Hourly-test.csv
```

## Main Results

The project shows how the selected hourly time series can be separated into trend, seasonal, and residual components.

The classical decomposition helps identify the main structure of the data. Residual analysis is then used to check whether the remaining noise still contains patterns.

After that, an ARIMA model is applied to the residual series to improve forecasting accuracy.

The final forecasts are evaluated against the first 24 observations of the test set and compared with mean, naive, and seasonal naive forecasting methods.

## Conclusion

This project demonstrates a full time series analysis workflow in Python. It starts with visual exploration, continues with classical decomposition and residual diagnostics, and ends with ARIMA forecasting and comparison against benchmark methods.

The analysis shows the importance of understanding trend and seasonality before applying forecasting models. It also shows why residual checking is necessary before trusting the final model.
