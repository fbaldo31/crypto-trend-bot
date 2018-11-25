"""
=======================================
Learning Bitcoin Market
=======================================

"""
from __future__ import print_function

# Author: Frederick Baldo
# License: ISC

import sys
from datetime import datetime
import csv

from sklearn.externals.six.moves.urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold

print(__doc__)

import json
from threading import Timer
from django.http import HttpResponse

from . import models

print(models)

# #############################################################################
# Retrieve the data from Internet

CRYPTO = "1" # Bitcoin has id = 1
API_URL = "https://api.coinmarketcap.com/v2/ticker/" + CRYPTO + "/?convert=EUR"
rates = []
one_day_rates = []
one_week_rates = []
one_hour_rates = []
current_rate = []
i = 0

# Get instant crypto rates
def pingApi():
    res = urlopen(API_URL)
    body = res.read()
    data = json.loads(body)
    #print(data)
    return data

# Put data in array
def getRates(index: int):
    data = pingApi()
    crypto = objectview(data)
    rate = crypto.__dict__['data']['quotes']['EUR']
    one_hour_rates.append(rate['percent_change_1h'])
    one_day_rates.append(rate['percent_change_24h'])
    one_week_rates.append(rate['percent_change_7d'])
    print(rate['percent_change_1h'])
    rates.append(crypto)
    index += 1
    return index

# Fetch Bitcoin data at regular intervals
def start():
    while (i < 10):
        timer = Timer(1.0, getRates, [i]) 
        timer.start()
        timer.join()

    print("End {rates.count}")

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
start()
