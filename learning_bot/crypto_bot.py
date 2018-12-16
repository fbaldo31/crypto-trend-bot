"""
=======================================
Learning Bitcoin Market
=======================================

"""
from __future__ import print_function

# Author: Frederick Baldo
# License: ISC

# import sys
# from datetime import datetime
# import csv

from sklearn.externals.six.moves.urllib.request import urlopen
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection

# from sklearn import cluster, covariance, manifold
from sklearn import preprocessing
import numpy as np

print(__doc__)

from os import environ
import json
from threading import Timer
from django.http import HttpResponse

from . import models

print(models)

# #############################################################################
# Retrieve the data from Internet

CRYPTO = "BTC" # Bitcoin
# API_KEY = "<YOUR_API_KEY>"
API_KEY = "da192bcd573229c4e5308b7f69842ab646aa4cdf649a8af4ce0e981e04e51c54"
LAST_YEAR_TREND = "https://min-api.cryptocompare.com/data/histoday?fsym=" + CRYPTO + "&tsym=EUR&limit=364&api_key=" + API_KEY + ""
CURRENT_BTC_RATE =  "https://min-api.cryptocompare.com/data/dayAvg?fsym=" + CRYPTO + "&tsym=EUR&api_key=" + API_KEY + ""
LAST_WEEK_TREND = "https://min-api.cryptocompare.com/data/histominute?fsym=" + CRYPTO + "&tsym=EUR&api_key=" + API_KEY + ""
rates = []
one_day_rates = []
one_week_rates = []
one_hour_rates = []
year_high = []
year_low = []
week_high = []
week_low = []
current_rate = []
i = 0

# Get instant crypto rates
def getStats(url: str):
    res = urlopen(url)
    body = res.read()
    data = json.loads(body)
    return data['Data']

def getCurrentValue(url: str):
    res = urlopen(url)
    body = res.read()
    data = json.loads(body)
    print(body)
    return data['EUR']

# Put data in array
def getRates():
    print("Get " + CRYPTO + " daily data from 1 year...")
    last_year_data = getStats(LAST_YEAR_TREND)
    print("Get " + CRYPTO + " data by minute from 1 week...")
    last_week_data = getStats(LAST_WEEK_TREND)

    print("Got data, preparing it ...")
    for item in last_year_data:
        item = objectview(item)
        print(item.__dict__['high'])
        # rates.append(item)
        year_high.append(item.__dict__['high'])
        year_low.append(item.__dict__['low'])

    for item in last_week_data:
        item = objectview(item)
        week_high.append(item.__dict__['high'])
        week_low.append(item.__dict__['low'])
        print(item.__dict__['high'])

# Fetch Bitcoin data at regular intervals
def start():
    getRates()
    year_stats = preprocessingData(year_high, year_low)
    week_stats = preprocessingData(week_high, week_low)
    print("Listen current market...")
    floodApi()
    
# Fetch current trend every minute
def floodApi():
    i = 0
    while (1 == 1):
        timer = Timer(60.0, getCurrentValue, [CURRENT_BTC_RATE]) 
        timer.start()
        timer.join()
        # print(timer)

# Make the preprocessing
# @see https://scikit-learn.org/stable/modules/preprocessing.html
def preprocessingData(high, low):
    print("Start preprocessing...")
    # X_train = np.array([one_hour_rates, one_day_rates, one_week_rates])
    X_train = np.array([high, low])

    X_scaled = preprocessing.scale(X_train)
    X_scaled.mean(axis=0)
    X_scaled.std(axis=0)
    print(X_scaled)
    # Scaling
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_train)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)

    print(X_train_minmax)
    return X_scaled


# Permit to access to object properties from array
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

# Launch app on startup
# start()
