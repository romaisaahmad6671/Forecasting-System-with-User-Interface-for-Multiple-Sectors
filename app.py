import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM


# Assuming the seasonal period is 24 hours (daily data)
seasonal_period = 24  # Define seasonal_period at a level accessible by all model choices


def load_model(model_path):
    if model_path.endswith('.pkl'):
        import pickle
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    elif model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    return model


# Sidebar for data upload and model selection
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv'])
if uploaded_file is None:
    st.sidebar.write("Please upload a file to get started.")
    st.stop()

data = pd.read_csv(uploaded_file)
data['time'] = pd.to_datetime(data['time'], utc=True)
# data.set_index('time', inplace=True)
target_column = 'total load actual'

model_choice = st.sidebar.selectbox(
    "Select Model", ['ARIMA', 'Prophet', 'LSTM', 'SARIMA', 'SVR', 'Simple Exponential Smoothing', 'ANN'])

# Show raw data
if st.checkbox('Show raw data'):
    st.write(data)

# ARIMA Specific Operations
if model_choice == 'ARIMA':
    if st.button('Plot ACF/PACF'):
        # Plot ACF and PACF
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        plot_acf(data[target_column], lags=40, ax=axes[0])
        plot_pacf(data[target_column], lags=40, ax=axes[1])
        st.pyplot(fig)

    if st.button('Forecast'):
        # Load model (assuming model is saved in the same directory and named 'arima_model.pkl')
        model = load_model('arima_model.pkl')
        forecast_steps = 100  # Set your forecast steps
        forecast = model.predict(start=0, end=forecast_steps-1)

        # Display forecast
        fig, ax = plt.subplots()
        ax.plot(data.index[:forecast_steps],
                data[target_column][:forecast_steps], label='Actual')
        ax.plot(data.index[:forecast_steps], forecast,
                label='Forecast', color='red')
        ax.set_title('Forecast vs Actuals')
        ax.legend()
        st.pyplot(fig)

        # Error Metrics
        rmse = np.sqrt(mean_squared_error(
            data[target_column][:forecast_steps], forecast))
        mae = mean_absolute_error(
            data[target_column][:forecast_steps], forecast)
        st.write(f'Root Mean Squared Error: {rmse}')
        st.write(f'Mean Absolute Error: {mae}')


elif model_choice == 'Prophet':
    # Split the data into training and test sets
    # Prepare the data for Prophet
    data['time'] = pd.to_datetime(data['time'], utc=True)
    data['time'] = data['time'].dt.date
    df_prophet = data[['time', 'total load actual']].rename(
        columns={'time': 'ds', 'total load actual': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    train_size = int(len(df_prophet) * 0.8)
    train, test = df_prophet.iloc[:train_size], df_prophet.iloc[train_size:]

    if st.button('Train Prophet Model'):
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        model.fit(train)

        # Save the model using pickle
        with open('prophet_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Load the model back from the file
        with open('prophet_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Create a dataframe to hold predictions, let's forecast the length of the test set
        future = loaded_model.make_future_dataframe(
            periods=len(test), freq='D')
        forecast = loaded_model.predict(future)

        # Align the actual test values with the predicted values
        test['yhat'] = forecast['yhat'].iloc[-len(test):].values

        # Calculate error metrics
        rmse_loaded = np.sqrt(mean_squared_error(test['y'], test['yhat']))
        mae_loaded = mean_absolute_error(test['y'], test['yhat'])
        mape_loaded = mean_absolute_percentage_error(test['y'], test['yhat'])

        st.write(f'Root Mean Squared Error (Prophet Model): {rmse_loaded}')
        st.write(f'Mean Absolute Error (Prophet Model): {mae_loaded}')
        st.write(
            f'Mean Absolute Percentage Error (Prophet Model): {mape_loaded}')

        # Plot the forecast
        fig1 = loaded_model.plot(forecast)
        plt.title('Forecast of Total Load Actual')
        st.pyplot(fig1)

        # Plot the forecast components
        fig2 = loaded_model.plot_components(forecast)
        st.pyplot(fig2)

elif model_choice == 'LSTM':
    if st.button('Train and Forecast using LSTM'):
        # Normalize data
        train_size = int(len(data) * 0.8)
        train, test = data.iloc[:train_size], data.iloc[train_size:]
        train_mean = train[target_column].mean()
        train_std = train[target_column].std()
        train_scaled = (train[target_column] - train_mean) / train_std
        test_scaled = (test[target_column] - train_mean) / train_std

        # Create dataset for LSTM
        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:(i + time_step)])
                Y.append(data[i + time_step])
            return np.array(X), np.array(Y)

        time_step = 10
        X_train, y_train = create_dataset(train_scaled.values, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True,
                                 input_shape=(time_step, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(X_train, y_train, batch_size=1, epochs=10)

        # Forecast
        X_test, y_test = create_dataset(test_scaled.values, time_step)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        test_predict = model.predict(X_test)
        test_predict = test_predict * train_std + train_mean
        y_test_actual = test[target_column].iloc[time_step:].values

        # Calculate error metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
        mae = mean_absolute_error(y_test_actual, test_predict)
        mape = mean_absolute_percentage_error(y_test_actual, test_predict)

        # Display forecast and metrics
        fig, ax = plt.subplots()
        ax.plot(test.index[time_step:], y_test_actual, label='Actual')
        ax.plot(test.index[time_step:], test_predict,
                label='Forecast', color='red')
        ax.legend()
        st.pyplot(fig)

        st.write(f'Root Mean Squared Error: {rmse}')
        st.write(f'Mean Absolute Error: {mae}')
        st.write(f'Mean Absolute Percentage Error: {mape}')


# Functionality for LSTM
elif model_choice == 'SARIMA':
    if st.button('Plot ACF/PACF'):
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        sm.graphics.tsa.plot_acf(
            data[target_column], lags=seasonal_period*2, ax=axes[0])
        sm.graphics.tsa.plot_pacf(
            data[target_column], lags=seasonal_period*2, ax=axes[1])
        st.pyplot(fig)

    if st.button('Forecast SARIMA'):
        # Define and fit the SARIMA model
        # Use the previously defined seasonal_period
        seasonal_order = (1, 1, 1, seasonal_period)
        model = SARIMAX(data[target_column], order=(
            1, 1, 1), seasonal_order=seasonal_order)
        model_fit = model.fit()
        forecast_steps = 100  # Set your forecast steps
        forecast = model_fit.predict(start=0, end=forecast_steps-1)

        # Plot forecast
        fig, ax = plt.subplots()
        ax.plot(data.index[:forecast_steps],
                data[target_column][:forecast_steps], label='Actual')
        ax.plot(data.index[:forecast_steps], forecast,
                label='Forecast', color='red')
        ax.set_title('Forecast vs Actuals')
        ax.legend()
        st.pyplot(fig)

        # Error Metrics
        rmse = np.sqrt(mean_squared_error(
            data[target_column][:forecast_steps], forecast))
        mae = mean_absolute_error(
            data[target_column][:forecast_steps], forecast)
        mape = mean_absolute_percentage_error(
            data[target_column][:forecast_steps], forecast)
        st.write(f'Root Mean Squared Error: {rmse}')
        st.write(f'Mean Absolute Error: {mae}')
        st.write(f'Mean Absolute Percentage Error: {mape}')

elif model_choice == 'SVR':
    data['time'] = pd.to_datetime(data['time'], utc=True)
    data.set_index('time', inplace=True)

    # Prepare the data for the model
    target_column = 'total load actual'
    feature_columns = ['total load forecast']  # Example of exogenous variable

    if target_column in data.columns and all(col in data.columns for col in feature_columns):
        data = data.dropna(subset=[target_column])

        X = data[feature_columns]
        y = data[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)

        # Standardize the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(
            y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(
            y_test.values.reshape(-1, 1)).flatten()

    # GridSearch for hyperparameters
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(
        SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_scaled)

    # Best parameters
    best_params = grid_search.best_params_
    best_svr = SVR(kernel=best_params['kernel'],
                   C=best_params['C'], gamma=best_params['gamma'])
    best_svr.fit(X_train_scaled, y_train_scaled)

    # Predictions
    y_train_pred_scaled = best_svr.predict(X_train_scaled)
    y_test_pred_scaled = best_svr.predict(X_test_scaled)

    # Invert scaling
    y_train_pred = scaler_y.inverse_transform(
        y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)).flatten()

    # Metrics
    rmse_loaded = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_loaded = mean_absolute_error(y_test, y_test_pred)
    mape_loaded = mean_absolute_percentage_error(y_test, y_test_pred)

    st.write(f'Root Mean Squared Error (SVR Model): {rmse_loaded}')
    st.write(f'Mean Absolute Error (SVR Model): {mae_loaded}')
    st.write(f'Mean Absolute Percentage Error (SVR Model): {mape_loaded}')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.index, y_test, label='Test')
    ax.plot(y_test.index, y_test_pred, label='Forecast', color='red')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel(target_column)
    ax.set_title(f'{target_column} Forecast vs Actual')
    st.pyplot(fig)


elif model_choice == 'SVR':
    ets_model_choice = st.sidebar.selectbox(
        "Select ETS Model",
        ['Simple Exponential Smoothing',
            "Holt's Linear Trend", "Holt-Winters Seasonal"]
    )
    df = pd.read_csv(uploaded_file)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.date
    df.set_index('time', inplace=True)

    data = df['total load actual']

    # Split the data into training and validation sets
    split_point = int(len(data) * 0.8)
    train, valid = data[:split_point], data[split_point:]

    if model_choice == 'Simple Exponential Smoothing':
        model = SimpleExpSmoothing(train).fit()
        title = 'Simple Exponential Smoothing Model'
    elif ets_model_choice == "Holt's Linear Trend":
        model = Holt(train).fit()
        title = "Holt's Linear Trend Model"
    else:
        model = ExponentialSmoothing(
            train, seasonal='add', seasonal_periods=12).fit()
        title = "Holt-Winters Seasonal Model"

    # Forecast
    predictions = model.forecast(len(valid))
    rmse = np.sqrt(mean_squared_error(valid, predictions))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train.index, train, label='Train')
    ax.plot(valid.index, valid, label='Valid')
    ax.plot(valid.index, predictions, label='Forecast')
    ax.set_title(f'{title} - RMSE: {rmse:.2f}')
    ax.legend()
    st.pyplot(fig)

    # Save the model
    st.sidebar.download_button(
        label="Download model as pickle",
        data=pickle.dumps(model),
        file_name=f'{title.lower().replace(" ", "_")}.pkl',
        mime='application/octet-stream'
    )

    # Display RMSE
    st.write(f'{title} RMSE: {rmse:.2f}')


elif model_choice == 'Simple Exponential Smoothing':
    ets_model_choice = st.sidebar.selectbox(
        "Select ETS Model",
        ['Simple Exponential Smoothing',
            "Holt's Linear Trend", "Holt-Winters Seasonal"]
    )

    if 'total load actual' in data.columns:
        data = data['total load actual']

        # Split the data into training and validation sets
        split_point = int(len(data) * 0.8)
        train, valid = data[:split_point], data[split_point:]

        if ets_model_choice == 'Simple Exponential Smoothing':
            model = SimpleExpSmoothing(train).fit()
            title = 'Simple Exponential Smoothing Model'
        elif ets_model_choice == "Holt's Linear Trend":
            model = Holt(train).fit()
            title = "Holt's Linear Trend Model"
        else:
            model = ExponentialSmoothing(
                train, seasonal='add', seasonal_periods=12).fit()
            title = "Holt-Winters Seasonal Model"

        # Forecast
        predictions = model.forecast(len(valid))
        rmse = np.sqrt(mean_squared_error(valid, predictions))

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train.index, train, label='Train')
        ax.plot(valid.index, valid, label='Valid')
        ax.plot(valid.index, predictions, label='Forecast')
        ax.set_title(f'{title} - RMSE: {rmse:.2f}')
        ax.legend()
        st.pyplot(fig)

        # Save the model
        st.sidebar.download_button(
            label="Download model as pickle",
            data=pickle.dumps(model),
            file_name=f'{title.lower().replace(" ", "_")}.pkl',
            mime='application/octet-stream'
        )

        # Display RMSE
        st.write(f'{title} RMSE: {rmse:.2f}')
    else:
        st.error(
            "The required column 'total load actual' is not in the uploaded file.")
elif model_choice == 'ANN':

    data = data['total load actual']
    # Ensure no NaN values
    data = data.dropna()

    # Splitting the data into training and test sets
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    # Normalize the data
    train_mean = train.mean()
    train_std = train.std()

    train_scaled = (train - train_mean) / train_std
    test_scaled = (test - train_mean) / train_std

    # Prepare the data for the ANN model
    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step)])
            Y.append(data[i + time_step])
        return np.array(X), np.array(Y)

    time_step = 10
    X_train, y_train = create_dataset(train_scaled.values, time_step)
    X_test, y_test = create_dataset(test_scaled.values, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Design the ANN model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)

    # Forecast with the model
    test_predict = model.predict(X_test)
    test_predict = test_predict.flatten() * train_std + train_mean

    # Ensure y_test is correctly aligned with test_predict
    y_test_actual = test.iloc[time_step:].values

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test.index[time_step:], y_test_actual, label='Test Actual')
    ax.plot(test.index[time_step:], test_predict,
            label='Forecast', color='red')
    ax.set_title(f'{data.name} Forecast vs Actual - RMSE: {rmse:.2f}')
    ax.legend()
    st.pyplot(fig)

    # Save the model
    st.sidebar.download_button(
        "Download ANN Model",
        data=model.save("ann_model.h5"),
        file_name="ann_model.h5"
    )

    # Display RMSE
    st.write(f'Root Mean Squared Error: {rmse:.2f}')

else:
    st.sidebar.write("Please upload a file to get started.")
    st.stop()
