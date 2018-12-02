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

CRYPTO = "BTC" # Bitcoin has id = 1
API_KEY = "<YOUR_API_KEY>"
# API_URL = "https://api.coinmarketcap.com/v2/ticker/" + CRYPTO + "/?convert=EUR"
API_URL = "https://min-api.cryptocompare.com/data/histoday?fsym=" + CRYPTO + "&tsym=EUR&limit=364&api_key=" + API_KEY + ""
BTC_RATE =  "https://min-api.cryptocompare.com/data/dayAvg?fsym=" + CRYPTO + "&tsym=EUR&api_key=" + API_KEY + ""

rates = []
one_day_rates = []
one_week_rates = []
one_hour_rates = []
high = []
low = []
current_rate = []
i = 0

# Get instant crypto rates
def lastYearStats(url: str):
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
    print("Get " + CRYPTO + " data from 1 year...")
    data = lastYearStats(API_URL)
    print("Got data, preparing it ...")
    for item in data:
        item = objectview(item)
        print(item.__dict__['high'])
        rates.append(item)
        high.append(item.__dict__['high'])
        low.append(item.__dict__['low'])
    # crypto = objectview(data)
    # rate = crypto.__dict__['data']['quotes']['EUR']
    # one_hour_rates.append(rate['percent_change_1h'])
    # one_day_rates.append(rate['percent_change_24h'])
    # one_week_rates.append(rate['percent_change_7d'])
    # print(rates)
    return rates

# Fetch Bitcoin data at regular intervals
def start():
    # i = 0
    # while (i < 10):
    #     timer = Timer(1.0, getRates, [i]) 
    #     timer.start()
    #     timer.join()
    #     i += 1
    data = getRates()
    preprocessingData()
    print("Listen current market...")
    flood()
    print("End")
    
# Fetch current trend every minute
def flood():
    i = 0
    while (1 == 1):
        timer = Timer(60.0, getCurrentValue, [BTC_RATE]) 
        timer.start()
        timer.join()
        print(timer)

# Make the preprocessing
# @see https://scikit-learn.org/stable/modules/preprocessing.html
def preprocessingData():
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
start()
