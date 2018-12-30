# https://activewizards.com/blog/bitcoin-price-forecasting-with-deep-learning-algorithms/

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras import initializers
from matplotlib import pyplot
from datetime import datetime
from matplotlib import pyplot as plt
import plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()
# %matplotlib inline
from sklearn.externals.six.moves.urllib.request import urlopen
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg
# Access to object properties from array
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
class Rate():
    time = 0
    open = 0.0
    high = 0.0
    low = 0.0
    close = 0.0
    volumeto = 0.0
    volumefrom = 0.0
# load the dataset
CRYPTO = "BTC" # Bitcoin
# API_KEY = "<YOUR_API_KEY>"
API_KEY = "da192bcd573229c4e5308b7f69842ab646aa4cdf649a8af4ce0e981e04e51c54"
LAST_YEAR_TREND = "https://min-api.cryptocompare.com/data/histoday?fsym=" + CRYPTO + "&tsym=EUR&limit=364&api_key=" + API_KEY + ""
ALL_RATES = []

def getStats(url):
    res = urlopen(url)
    body = res.read()
    data = json.loads(body)
    return data['Data']

def getRates():
    print("Get " + CRYPTO + " daily data from 1 year...")
    last_year_data = getStats(LAST_YEAR_TREND)

    print("Got data, preparing it ...")
    for item in last_year_data:
        rate = Rate()
        item = objectview(item)
        rate.high = item.__dict__['high']
        rate.low = item.__dict__['low']
        rate.time = item.__dict__['time']
        print(item.__dict__['close'])
        ALL_RATES.append(rate)
    return ALL_RATES

# data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv')
jsonData = getStats(LAST_YEAR_TREND)
# data = pd.read_json('data.json') # , typ='frame', orient='records', convert_dates=True)
data = pd.DataFrame(jsonData)
# data.isnull().values.any()
data.head(10)

# Transform data to get the average price grouped by the day
data['date'] = pd.to_datetime(data['time'],unit='s').dt.date
group = data.groupby('date')
Daily_Price = group['close'].mean()

Daily_Price.head()

Daily_Price.tail()



# We need to split our dataset
from datetime import date
d0 = date(2016, 1, 1)
d1 = date(2017, 10, 15)
delta = d1 - d0
days_look = delta.days + 1
print(days_look)

d0 = date(2017, 8, 21)
d1 = date(2017, 10, 20)
delta = d1 - d0
days_from_train = delta.days + 1
print(days_from_train)

d0 = date(2017, 10, 15)
d1 = date(2017, 10, 20)
delta = d1 - d0
days_from_end = delta.days + 1
print(days_from_end)

# Now we are splitting our data into the train and test set:


df_train= Daily_Price[len(Daily_Price)-days_look-days_from_end:len(Daily_Price)-days_from_train]
df_test= Daily_Price[len(Daily_Price)-days_from_train:]

print(len(df_train), len(df_test))

#
# Exploratory Data Analysis 
#


working_data = [df_train, df_test]
working_data = pd.concat(working_data)

working_data = working_data.reset_index()
working_data['date'] = pd.to_datetime(working_data['date'])
working_data = working_data.set_index('date')

s = sm.tsa.seasonal_decompose(working_data.close.values, freq=60)

trace1 = go.Scatter(x = np.arange(0, len(s.trend), 1),y = s.trend,mode = 'lines',name = 'Trend',
    line = dict(color = ('rgb(244, 146, 65)'), width = 4))
trace2 = go.Scatter(x = np.arange(0, len(s.seasonal), 1),y = s.seasonal,mode = 'lines',name = 'Seasonal',
    line = dict(color = ('rgb(66, 244, 155)'), width = 2))

trace3 = go.Scatter(x = np.arange(0, len(s.resid), 1),y = s.resid,mode = 'lines',name = 'Residual',
    line = dict(color = ('rgb(209, 244, 66)'), width = 2))

trace4 = go.Scatter(x = np.arange(0, len(s.observed), 1),y = s.observed,mode = 'lines',name = 'Observed',
    line = dict(color = ('rgb(66, 134, 244)'), width = 2))

data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Seasonal decomposition', xaxis = dict(title = 'Time'), yaxis = dict(title = 'Price, USD'))

sd_data = data
sd_layout = layout

def get_seasonal_decompose():
    return json.dumps({ 'data': sd_data, 'layout': sd_layout }, cls=plotly.utils.PlotlyJSONEncoder)

# fig = dict(data=data, layout=layout)
# py.iplot(fig, filename='seasonal_decomposition', image='png')

# # examination of the autocorrelation
# plt.figure(figsize=(15,7))
# ax = plt.subplot(211)
# sm.graphics.tsa.plot_acf(working_data.close.values.squeeze(), lags=48, ax=ax)
# ax = plt.subplot(212)
# sm.graphics.tsa.plot_pacf(working_data.close.values.squeeze(), lags=48, ax=ax)
# plt.tight_layout()
# plt.show()

# # Now we need to recover our df_train and df_test datasets:
# df_train = working_data[:-60]
# df_test = working_data[-60:]

# #
# # Data preparation
# #

# def create_lookback(dataset, look_back=1):
#     X, Y = [], []
#     for i in range(len(dataset) - look_back):
#         a = dataset[i:(i + look_back), 0]
#         X.append(a)
#         Y.append(dataset[i + look_back, 0])
#     return np.array(X), np.array(Y)

# from sklearn.preprocessing import MinMaxScaler

# training_set = df_train.values
# training_set = np.reshape(training_set, (len(training_set), 1))
# test_set = df_test.values
# test_set = np.reshape(test_set, (len(test_set), 1))

# #scale datasets
# scaler = MinMaxScaler()
# training_set = scaler.fit_transform(training_set)
# test_set = scaler.transform(test_set)

# # create datasets which are suitable for time series forecasting
# look_back = 1
# X_train, Y_train = create_lookback(training_set, look_back)
# X_test, Y_test = create_lookback(test_set, look_back)

#  # reshape datasets so that they will be ok for the requirements of the LSTM model in Keras
# X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))


# # 
# # Training 2-layers LSTM Neural Network  
# #

# # initialize sequential model, add 2 stacked LSTM layers and densely connected output neuron
# model = Sequential()
# model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(LSTM(256))
# model.add(Dense(1))

# # compile and fit the model
# model.compile(loss='mean_squared_error', optimizer='adam')
# history = model.fit(X_train, Y_train, epochs=100, batch_size=16, shuffle=False,
#                     validation_data=(X_test, Y_test),
#                     callbacks = [EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1)])


# trace1 = go.Scatter(
#     x = np.arange(0, len(history.history['loss']), 1),
#     y = history.history['loss'],
#     mode = 'lines',
#     name = 'Train loss',
#     line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')
# )
# trace2 = go.Scatter(
#     x = np.arange(0, len(history.history['val_loss']), 1),
#     y = history.history['val_loss'],
#     mode = 'lines',
#     name = 'Test loss',
#     line = dict(color=('rgb(244, 146, 65)'), width=2)
# )

# data = [trace1, trace2]
# layout = dict(title = 'Train and Test Loss during training',
#               xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))
# fig = dict(data=data, layout=layout)
# py.iplot(fig, filename='training_process', image='png')


# # add one additional data point to align shapes of the predictions and true labels
# X_test = np.append(X_test, scaler.transform(working_data.iloc[-1][0]))
# X_test = np.reshape(X_test, (len(X_test), 1, 1))

# # get predictions and then make some transformations to be able to calculate RMSE properly in USD
# prediction = model.predict(X_test)
# prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
# Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
# prediction2_inverse = np.array(prediction_inverse[:,0][1:])
# Y_test2_inverse = np.array(Y_test_inverse[:,0])


# trace1 = go.Scatter(
#     x = np.arange(0, len(prediction2_inverse), 1),
#     y = prediction2_inverse,
#     mode = 'lines',
#     name = 'Predicted labels',
#     line = dict(color=('rgb(244, 146, 65)'), width=2)
# )
# trace2 = go.Scatter(
#     x = np.arange(0, len(Y_test2_inverse), 1),
#     y = Y_test2_inverse,
#     mode = 'lines',
#     name = 'True labels',
#     line = dict(color=('rgb(66, 244, 155)'), width=2)
# )

# data = [trace1, trace2]
# layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted',
#              xaxis = dict(title = 'Day number'), yaxis = dict(title = 'Price, USD'))
# fig = dict(data=data, layout=layout)
# py.iplot(fig, filename='results_demonstrating0', image='png')

# RMSE = sqrt(mean_squared_error(Y_test2_inverse, prediction2_inverse))
# print('Test RMSE: %.3f' % RMSE)


# Test_Dates = Daily_Price[len(Daily_Price)-days_from_train:].index

# trace1 = go.Scatter(x=Test_Dates, y=Y_test2_inverse, name= 'Actual Price',
#                    line = dict(color = ('rgb(66, 244, 155)'),width = 2))
# trace2 = go.Scatter(x=Test_Dates, y=prediction2_inverse, name= 'Predicted Price',
#                    line = dict(color = ('rgb(244, 146, 65)'),width = 2))
# data = [trace1, trace2]
# layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
#              xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
# fig = dict(data=data, layout=layout)
# py.iplot(fig, filename='results_demonstrating1', image='png')


# # This function prepares random train/test split, 
# # scales data with MinMaxScaler, create time series labels (Y)
# def get_split(working_data, n_train, n_test, look_back = 1):
#     # get a point from which we start to take train dataset and after it - test dataset
#     start_point = randint(0, (len(working_data)-n_test-n_train))
#     df_train = working_data[start_point:start_point+n_train]
#     df_test = working_data[start_point+n_train:start_point+n_train+n_test]

#     training_set = df_train.values
#     training_set = np.reshape(training_set, (len(training_set), 1))
#     test_set = df_test.values
#     test_set = np.reshape(test_set, (len(test_set), 1))

#     # scale datasets
#     scaler_cv = MinMaxScaler()
#     training_set = scaler_cv.fit_transform(training_set)
#     test_set = scaler_cv.transform(test_set)

#     # create datasets which are suitable for time series forecasting
#     X_train, Y_train = create_lookback(training_set, look_back)
#     X_test, Y_test = create_lookback(test_set, look_back)

#     # reshape datasets so that they will be ok for the requirements of the models in Keras
#     X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
#     X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

#     return X_train, Y_train, X_test, Y_test, scaler_cv, start_point

# # This function takes datasets from the previous function as input and train model using these datasets
# def train_model(X_train, Y_train, X_test, Y_test):
#     # initialize sequential model, add bidirectional LSTM layer and densely connected output neuron
#     model = Sequential()
#     model.add(GRU(256, input_shape=(X_train.shape[1], X_train.shape[2])))
#     model.add(Dense(1))

#     # compile and fit the model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.fit(X_train, Y_train, epochs = 100, batch_size = 16, shuffle = False,
#                     validation_data=(X_test, Y_test), verbose=0,
#                     callbacks = [EarlyStopping(monitor='val_loss',min_delta=5e-5,patience=20,verbose=0)])
#     return model

# # This function uses trained model and test dataset to calculate RMSE
# def get_rmse(model, X_test, Y_test, scaler, start_point, working_data, n_train):
#     # add one additional data point to align shapes of the predictions and true labels
#     X_test = np.append(X_test, scaler.transform(working_data.iloc[start_point+n_train+len(X_test)][0]))
#     X_test = np.reshape(X_test, (len(X_test), 1, 1))

#     # get predictions and then make some transformations to be able to calculate RMSE properly in USD
#     prediction = model.predict(X_test)
#     prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
#     Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
#     prediction2_inverse = np.array(prediction_inverse[:,0][1:])
#     Y_test2_inverse = np.array(Y_test_inverse[:,0])

#     #calculate

#     RMSE = sqrt(mean_squared_error(Y_test2_inverse, prediction2_inverse))
#     return RMSE, prediction2_inverse

# # The function below uses all three previous functions to build workflow of calculations and return RMSE and predictions of the model.
# def workflow(working_data, get_split, train_model, get_rmse,n_train = 250,n_test = 50,look_back = 1):
#     X_train, Y_train, X_test, Y_test, scaler, start_point = get_split(working_data, n_train, n_test)
#     model = train_model(X_train, Y_train, X_test, Y_test)
#     RMSE, predictions = get_rmse(model, X_test, Y_test, scaler, start_point, working_data, n_train)
#     return RMSE, predictions
# RMSE, predictions = workflow(working_data, get_split, train_model, get_rmse, n_train = 600,n_test = 60)
# print('Test GRU model RMSE: %.3f' % RMSE)

# # This function is used to repeat the workflow ten times and to calculate average RMSE
# def cross_validate(working_data,get_split,train_model,get_rmse,workflow,n_train = 250,n_test = 50,look_back = 1):
#     rmse_list = []
#     for i in range(10):
#         print('Iteration:', i+1)
#         RMSE, _ = workflow(working_data, get_split, train_model, get_rmse, n_train, n_test, look_back)
#         rmse_list.append(RMSE)
#         print('Test RMSE: %.3f' % RMSE)
#     mean_rmse = np.mean(rmse_list)
#     return mean_rmse, rmse_list


# mean_rmse, rmse_list = cross_validate(working_data, get_split, train_model, get_rmse, workflow)
# print('Average RMSE: ', mean_rmse)
# print('RMSE list:', rmse_list)

# predictions_new = predictions - mean_rmse

# RMSE_new = sqrt(mean_squared_error(Y_test2_inverse, predictions_new))
# print('Test GRU model RMSE_new: %.3f' % RMSE_new)

# trace1 = go.Scatter(x=Test_Dates, y=Y_test2_inverse, name= 'Actual Price',
#                    line = dict(color = ('rgb(66, 244, 155)'),width = 2))
# trace2 = go.Scatter(x=Test_Dates, y=predictions_new, name= 'Predicted Price',
#                    line = dict(color = ('rgb(244, 146, 65)'),width = 2))
# data = [trace1, trace2]
# layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
#              xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
# fig = dict(data=data, layout=layout)
# py.iplot(fig, filename='results_demonstrating2')
# plt.show()

# # Let's calculate a symmetric mean absolute percentage error ( SMAPE). It will show how good our predictions are in percentage. We define function symmetric_mean_absolute_percentage_error, which will perform all necessary calculations.
# def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon = 1e-8):
#     return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred))/2 + epsilon)) * 100

# SMAPE = symmetric_mean_absolute_percentage_error(Y_test2_inverse, predictions_new)

# print('Test SMAPE (percentage): %.3f' % SMAPE)