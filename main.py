import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime, timedelta
import datetime as dt

def fb_anomaly_df(df):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    anomaly_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    anomaly_df = pd.merge(df,anomaly_df, on='ds')

    anomaly_df['Type'] = ''
    anomaly_df['distance'] = 0.0
    for i in range(anomaly_df.shape[0]):
        if (anomaly_df['y'][i] > anomaly_df['yhat_upper'][i]):
            anomaly_df['Type'][i] = 'U'
            anomaly_df['distance'][i] = anomaly_df['y'][i] - anomaly_df['yhat_upper'][i]
        elif (anomaly_df['y'][i] < anomaly_df['yhat_lower'][i]): 
            anomaly_df['Type'][i] = 'L'
            anomaly_df['distance'][i] = anomaly_df['yhat_lower'][i] - anomaly_df['y'][i]

    #fig1 = m.plot(forecast)
    #plot_plotly(m, forecast)
    return anomaly_df

def anomalies_display(data):
    
    anomaly_df= fb_anomaly_df(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anomaly_df.ds, y=anomaly_df.yhat_upper, mode='lines', name='Upper baseline'))
    fig.add_trace(go.Scatter(x=anomaly_df.ds, y=anomaly_df.y, mode='markers', name='Actual Value'))
    fig.add_trace(go.Scatter(x=anomaly_df.ds, y=anomaly_df.yhat, mode='lines', name='Predicted value'))
    fig.add_trace(go.Scatter(x=anomaly_df.ds, y=anomaly_df.yhat_lower, mode='lines', name='Lower baseline'))
    fig.update_layout(
        title="Time series Forecasting using fbProphet",
        xaxis_title="Date-Time",
        yaxis_title="Values",
        legend_title="Legend",
    )
    fig.show()
    anomaly_df = anomaly_df[anomaly_df['Type']!= '']
    return anomaly_df


end_date = datetime.now()
start_date = end_date - timedelta(days=6*30)
stock_data = yf.download(str(input("Stock name: ")), start=start_date, end=end_date)
print(stock_data.shape)
stock_data.tail()

fig = px.line(stock_data,  y="Close", title='Stock price chart')


close_prices = stock_data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs= int(input("epoch: ")))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)

data = stock_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
train.reset_index(inplace = True)
validation.reset_index(inplace = True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.Date, y=train.Close, mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=validation.Date, y=validation.Close, mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=validation.Date, y=validation.Predictions, mode='lines', name='Predicted price'))
fig.update_layout(
        title="Time series Forecasting using LSTM",
        xaxis_title="Date-Time",
        yaxis_title="Values",
        legend_title="Legend",
)
fig.show()
