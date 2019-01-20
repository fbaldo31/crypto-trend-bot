import os
import json
import plotly
import numpy as np
import pandas as pd
from math import sqrt
from random import randint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras import initializers
from server.settings import BASE_DIR
from django.http import HttpResponse

"""
ONLINE MODE
"""
# load the dataset from internet
CRYPTO = "BTC" # Bitcoin
API_KEY = "<YOUR_API_KEY>"
LAST_YEAR_TREND = "https://min-api.cryptocompare.com/data/histoday?fsym=" + CRYPTO + "&tsym=EUR&limit=364&api_key=" + API_KEY + ""
HISTORIC_BTC_DATA_FILE = os.path.join(BASE_DIR, 'data.csv')

def getStats(url):
    res = urlopen(url)
    body = res.read()
    data = json.loads(body)
    return data['Data']


def getData(mode):
    if mode == 'url':
        jsonData = getStats(LAST_YEAR_TREND)
        data = pd.DataFrame(jsonData)
        data.isnull().values.any()
        data.head(10)
        # Transform data to get the average price grouped by the day
        data['date'] = pd.to_datetime(data['time'],unit='s').dt.date
        group = data.groupby('date')
        Daily_Price = group['close'].mean()
        return Daily_Price

    elif mode == 'file':
        # Load dataset from local file
        data = pd.read_csv(HISTORIC_BTC_DATA_FILE)
        data.isnull().values.any()
        data.head(10)
        # Transform data to get the average price grouped by the day
        data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date
        group = data.groupby('date')
        Daily_Price = group['Weighted_Price'].mean()
        return Daily_Price
    else:
        return 'Bad mode, allow values are: url | file'

def prepareData(mode, trainStart, trainEnd, testS, testE):
    Daily_Price = getData(mode)
    Daily_Price.isnull().values.any()
    Daily_Price.head()
    Daily_Price.tail()
    
    # We need to split our dataset
    from datetime import date
    d0 = date(int(trainStart.split('-')[0]), int(trainStart.split('-')[1]), int(trainStart.split('-')[2])) # date(2016, 1, 1)
    d1 = date(int(trainEnd.split('-')[0]), int(trainEnd.split('-')[1]), int(trainEnd.split('-')[2])) # date(2018, 31, 15)
    delta = d1 - d0
    days_look = delta.days + 1
    print(days_look)

    d0 = date(int(testS.split('-')[0]), int(testS.split('-')[1]), int(testS.split('-')[2])) # date(2017, 8, 21)
    d1 = date(int(testE.split('-')[0]), int(testE.split('-')[1]), int(testE.split('-')[2])) # date(2017, 10, 20)
    delta = d1 - d0
    days_from_train = delta.days + 1
    print(days_from_train)

    d0 = date(int(trainEnd.split('-')[0]), int(trainEnd.split('-')[1]), int(trainEnd.split('-')[2]))
    d1 = date(int(testE.split('-')[0]), int(testE.split('-')[1]), int(testE.split('-')[2]))
    delta = d1 - d0
    days_from_end = delta.days + 1
    print(days_from_end)

    df_train= Daily_Price[len(Daily_Price)-days_look-days_from_end:len(Daily_Price)-days_from_train]
    df_test= Daily_Price[len(Daily_Price)-days_from_train:]

    print(len(df_train), len(df_test))

    return [df_train, df_test], Daily_Price, days_from_train

#
# Exploratory Data Analysis 
#
def getWorkingData(mode, trainStart, trainEnd, testS, testE):    
    working_data, Daily_Price, days_from_train = prepareData(mode, trainStart, trainEnd, testS, testE)
    working_data = pd.concat(working_data)

    working_data = working_data.reset_index()
    working_data['date'] = pd.to_datetime(working_data['date'])
    working_data = working_data.set_index('date')
    return working_data, Daily_Price, days_from_train

#
# Data preparation
#
def create_lookback(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# This function prepares random train/test split, 
# scales data with MinMaxScaler, create time series labels (Y)
def get_split(working_data, n_train, n_test, look_back = 1):
    # get a point from which we start to take train dataset and after it - test dataset
    start_point = randint(0, (len(working_data))) # -n_test-n_train))
    df_train = working_data[start_point:start_point+n_train]
    df_test = working_data[start_point+n_train:start_point+n_train+n_test]

    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    test_set = df_test.values
    test_set = np.reshape(test_set, (len(test_set), 1))

    # scale datasets
    scaler_cv = MinMaxScaler()
    training_set = scaler_cv.fit_transform(training_set)

    print(test_set)
    # Error management: Sometimes the array is empty
    if (len(test_set) == 0):
        raise ValueError("Something went wrong, please try again.")
  
    # Array has values
    test_set = scaler_cv.transform(test_set)
        
    # create datasets which are suitable for time series forecasting
    X_train, Y_train = create_lookback(training_set, look_back)
    X_test, Y_test = create_lookback(test_set, look_back)

    # reshape datasets so that they will be ok for the requirements of the models in Keras
    X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

    return X_train, Y_train, X_test, Y_test, scaler_cv, start_point

# This function takes datasets from the previous function as input and train model using these datasets
def train_model(X_train, Y_train, X_test, Y_test):
    # initialize sequential model, add bidirectional LSTM layer and densely connected output neuron
    model = Sequential()
    model.add(GRU(256, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))

    # compile and fit the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs = 100, batch_size = 16, shuffle = False,
                    validation_data=(X_test, Y_test), verbose=0,
                    callbacks = [EarlyStopping(monitor='val_loss',min_delta=5e-5,patience=20,verbose=0)])
    return model


def mse(x1, x2, axis=0):
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1-x2)**2, axis=axis)

def rmse(x1, x2, axis=0):
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))

# This function uses trained model and test dataset to calculate RMSE
def get_rmse(model, X_test, Y_test, scaler, start_point, working_data, n_train):
    # try fix | start_point+n_train+
    # arg = start_point+n_train+len(X_test)
    # print('arg', arg)
    # test = working_data.iloc[start_point+n_train+len(X_test)] # [0]

    # print('test', test)

    # transfomed = scaler.transform(working_data)

    # print('transfomed', transfomed)

    # # add one additional data point to align shapes of the predictions and true labels
    # X_test = np.append(X_test, transfomed)
    # print(X_test)
    # X_test = np.reshape(X_test, (len(X_test), 1, 1))
    # print(X_test)

    # get predictions and then make some transformations to be able to calculate RMSE properly in USD
    prediction = model.predict(X_test)
    prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
    
    Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
    prediction2_inverse = np.array(prediction_inverse[:,0][1:])
    Y_test2_inverse = np.array(Y_test_inverse[:,0])
    print('Prediction2', len(prediction2_inverse))
    # In case of the 2 arrays have not the same length
    Y_test2_inverse = Y_test2_inverse[0:len(prediction2_inverse)]
    print('Y_test2_inverse', len(Y_test2_inverse))
    #calculate
    RMSE = sqrt(mean_squared_error(Y_test2_inverse, prediction2_inverse))
    return RMSE, prediction2_inverse, Y_test2_inverse

# The function below uses all three previous functions to build workflow of calculations and return RMSE and predictions of the model.
def workflow(working_data, get_split, train_model, get_rmse,n_train = 250,n_test = 50,look_back = 1):
    X_train, Y_train, X_test, Y_test, scaler, start_point = get_split(working_data, n_train, n_test)
    model = train_model(X_train, Y_train, X_test, Y_test)
    RMSE, predictions, Y_test2_inverse = get_rmse(model, X_test, Y_test, scaler, start_point, working_data, n_train)
    return RMSE, predictions, Y_test2_inverse


# This function is used to repeat the workflow ten times and to calculate average RMSE
def cross_validate(working_data,get_split,train_model,get_rmse,workflow,n_train = 250,n_test = 50,look_back = 1):
    rmse_list = []
    for i in range(10): #10
        print('Iteration:', i+1)
        RMSE, _, x = workflow(working_data, get_split, train_model, get_rmse, n_train, n_test, look_back)
        rmse_list.append(RMSE)
        print('Test RMSE: %.3f' % RMSE)
    mean_rmse = np.mean(rmse_list)
    return mean_rmse, rmse_list

def runPipeline(*args):
    print(args)
    working_data, Daily_Price, days_from_train = getWorkingData('file', args[0], args[1], args[2], args[3])
    RMSE, predictions, Y_test2_inverse = workflow(working_data, get_split, train_model, get_rmse, n_train = 100,n_test = 60)
    print('Test GRU model RMSE: %.3f' % RMSE) 
    mean_rmse, rmse_list = cross_validate(working_data, get_split, train_model, get_rmse, workflow)
    print('Average RMSE: ', mean_rmse)
    print('RMSE list:', rmse_list)

    predictions_new = predictions - mean_rmse

    RMSE_new = sqrt(mean_squared_error(Y_test2_inverse, predictions_new))
    print('Test GRU model RMSE_new: %.3f' % RMSE_new)

    Test_Dates = Daily_Price[len(Daily_Price)-days_from_train:].index

    trace1 = go.Scatter(x=Test_Dates, y=Y_test2_inverse, name= 'Actual Price',
                       line = dict(color = ('rgb(66, 244, 155)'),width = 2))
    trace2 = go.Scatter(x=Test_Dates, y=predictions_new, name= 'Predicted Price',
                       line = dict(color = ('rgb(244, 146, 65)'),width = 2))
    data = [trace1, trace2]
    layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
                 xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))

    # Let's calculate a symmetric mean absolute percentage error ( SMAPE). It will show how good our predictions are in percentage. We define function symmetric_mean_absolute_percentage_error, which will perform all necessary calculations.
    def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon = 1e-8):
        return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred))/2 + epsilon)) * 100

    SMAPE = symmetric_mean_absolute_percentage_error(Y_test2_inverse, predictions_new)

    report = { 'testRMSE': RMSE, 'averageRMSE': mean_rmse, 'rmseList': rmse_list, 'newRMSE': RMSE_new, 'SMAPE': SMAPE };

    print('Test SMAPE (percentage): %.3f' % SMAPE)
    return json.dumps({ 'data': data, 'layout': layout, 'report': report }, cls=plotly.utils.PlotlyJSONEncoder)