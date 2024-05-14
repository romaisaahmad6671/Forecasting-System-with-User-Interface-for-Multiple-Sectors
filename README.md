# Forecasting-System-with-User-Interface-for-Multiple-Sectors

## Project Overview
This project develops a comprehensive forecasting system that implements and compares various time series models such as ARIMA, ANN, and Hybrid ARIMA-ANN across multiple sectors, including finance, energy, and environment. It features a user-friendly front-end interface for data visualization and forecast interaction.

## Data Source
For dataset we have used the energy sector that tells the hourly energy consumption data.

## Preprocessing Steps
- **Cleaning** : Identified and dropped the missing values.
- **Normalization** : Scale the data to a uniform range.
-  **Stationarization** : Apply differencing and logarithmic transformations as necessary to achieve stationarity.

## Models

### ARIMA
- ARIMA (Autoregressive Integrated Moving Average) is utilized to model and forecast time series data that shows levels of non-stationarity or seasonal patterns.
- First we calculated the Augmented Dickey-Fuller test for stationarity. After that we plotted the ACF/PACF plots for parameter estimation. Then we fit the model. After that we did forecasting and plotted it with the original values.
- **RMSE** = 0.3491319911092562
- **MAE** = 0.29128803372622897
- **MAPE** = 5.690330650121977

### ANN 
- The ANN model in this project employs Long Short-Term Memory (LSTM) layers to forecast time series data effectively. Data preparation involves scaling and transforming the data into sequences that serve as input to the neural network. 
- The model architecture includes two LSTM layers designed to capture temporal dependencies, followed by dense layers for output generation. The model is compiled with the Adam optimizer and trained using mean squared error loss.
-  **RMSE** = 048
-  **MAE** = 0.033
-  **MAPE** = 0.251

### SARIMA (Seasonal ARIMA)
- The SARIMA model in this project applies sophisticated statistical techniques to model and forecast seasonal time series data. The Augmented Dickey-Fuller (ADF) test is first used to ensure data stationarity, which is crucial for the accuracy of the model.
- Using Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots, the model parameters are identified and configured to account for both non-seasonal and seasonal influences, optimizing the model's predictive capabilities.
- The model fitting and evaluation involve training on historical data, followed by serialization for deployment and subsequent forecasting, ensuring model reusability and consistency in predictions.
- **RMSE**: 0.5519
- **MAE**: 0.5071
- **MAPE**: 5.6231

### Exponential Smoothing (ETS)
- The ETS models in this project are employed for their ability to forecast based on weighted averages of past observations, with the weights decaying exponentially over time. The models include Simple Exponential Smoothing for level data, Holt’s Linear Trend Model to address data with trends, and Holt-Winters Seasonal Model for handling both trends and seasonality.
- These models are tailored to the specific characteristics of the time series data from the "new_energy_data.csv" file, making appropriate adjustments for the smoothing parameters based on the observed data patterns.
- **Simple Exponential Smoothing RMSE**: 0.36
- **Holt's Linear Trend Model RMSE**: 0.44
- **Holt-Winters Seasonal Model RMSE**: 0.31

### Prophet
- Prophet is utilized in this project for its robustness in handling time series data that displays strong seasonal effects and holiday impacts. The model processes data that has been prepared to ensure all date stamps are correctly formatted and null values handled to avoid any parsing errors.
- The model leverages daily, weekly, and yearly seasonality components to forecast the 'total load actual', which is critical for predicting energy demands accurately.
- Post-model fitting, the forecast is extended into the future, aligning predicted values with actual data to evaluate performance comprehensively.
- **Root Mean Squared Error (RMSE)**: 1.637
- **Mean Absolute Error (MAE)**: 1.476
- **Mean Absolute Percentage Error (MAPE)**: 13.409

### Support Vector Regression (SVR)
- SVR is utilized in this project for its capability to handle non-linear relationships by using kernel functions. The feature set includes 'total load forecast', and the target is 'total load actual', making it highly suitable for regression analysis in time series forecasting.
- The data is standardized using `StandardScaler` to optimize performance, as SVR can be sensitive to the scale of input data.
- The model parameters are optimized using `GridSearchCV` to find the best combination of kernel type, C (regularization parameter), and gamma (kernel coefficient), ensuring the best fit to the historical data.
- **Root Mean Squared Error (RMSE)**: 0.0259
- **Mean Absolute Error (MAE)**: 0.0218
- **Mean Absolute Percentage Error (MAPE)**: 2.3749

### Long Short-Term Memory (LSTM)
- The LSTM model in this project utilizes layers specifically designed for time series forecasting by capturing long-term dependencies in data sequences. The model architecture comprises LSTM layers followed by a dense output layer, optimized using the Adam optimizer.
- The dataset, featuring time-indexed data, is preprocessed through normalization using `StandardScaler` to scale both features and target variables, enhancing model training efficiency.
- The model is trained over 20 epochs with a batch size of 72, showing progressive reduction in loss, indicating effective learning.
- Performance Metrics:
  - RMSE = 0.030
  - MAE = 0.028
  - MAPE = 0.183


### Time Series Forecasting Application

This Python-based application is designed to perform sophisticated time series analysis and forecasting using multiple statistical and machine learning models. Users can choose among ARIMA, SARIMA, Prophet, LSTM, SVR, and various exponential smoothing methods, depending on their data and forecasting needs.

#### Features:
- **Model Selection**: Users can select from a variety of models through a streamlined sidebar interface.
- **Data Handling**: The application supports CSV data uploads, allowing for easy integration and manipulation of user-specific time series data.
- **Interactive Forecasting**: Engage with real-time predictions and model tuning with interactive buttons directly within the app.
- **Visualization**: Dynamically generated plots display both the forecasts and actual data, offering clear visual insights into model performance.

#### Metrics and Validation:
- The application calculates and displays key performance metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and MAPE (Mean Absolute Percentage Error), aiding in the evaluation of model accuracy.

#### Customizability:
- Users have the flexibility to adjust model parameters and preprocess data to enhance forecast accuracy, tailored to specific needs.

### Installation and Setup
1. Clone the repository.
2. Ensure all dependencies are installed by running `pip install -r requirements.txt`.
3. Execute the Streamlit app using `streamlit run app.py`.

### Technologies Used:
- **Streamlit** for the web interface.
- **Pandas** and **NumPy** for data manipulation.
- **Matplotlib** and **Seaborn** for data visualization.
- **SciKit-Learn**, **Statsmodels**, and **TensorFlow** for modeling and forecasts.


## Contribution
- Romaisa Ahmad(21i-1702) has done the preprocessing, and has applied the models ARIMA, SARIMA, SVR, ANN.
- For this project Farhan Javaid(21i-1671) has created the UI, and has applied the models Prophet, ETS, LSTM.

