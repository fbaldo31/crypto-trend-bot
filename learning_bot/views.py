# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import Template, Context

#
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
import sys
import datetime
import numpy as np
from PIL import Image
import urllib

# Init Variables
btc_img = "http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png"
eth_img = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png"

class crypto():
    def __init__(self, currency):
        # get market info for bitcoin from the start of 2016 to the current day
        self.market_info = pd.read_html("https://coinmarketcap.com/currencies/"+currency+"/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
        # convert the date string to the correct date format
        self.market_info.assign(Date=pd.to_datetime(self.market_info['Date']))
        
        if currency == "bitcoin":
            img = btc_img
            pass
        else:
            img = eth_img
            pass
        self.img = urllib.request.urlopen(img)

# Get data on init
btc = crypto('bitcoin')
# when Volume is equal to '-' convert it to 0
btc.market_info.loc[btc.market_info['Volume']=="-",'Volume']=0
# convert to int
btc.market_info['Volume'] = btc.market_info['Volume'].astype('int64')
btc.market_info.columns = btc.market_info.columns.str.replace("*", "")
btc.market_info.head()

eth = crypto('ethereum')
# this will remove those asterisks
eth.market_info.columns = eth.market_info.columns.str.replace("*", "")
# look at the first few rows
eth.market_info.head()


def prepare():
    market_info = pd.merge(btc.market_info,eth.market_info, on=['Date'])
    market_info = market_info[market_info['Date']>='2017-01-01']
    for coins in ['bt_', 'eth_']: 
        kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
        market_info = market_info.assign(**kwargs)
    market_info.head()
    return market_info

def prepareLtsm(market_info):
    for coins in ['bt_', 'eth_']: 
        kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
            coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
    market_info = market_info.assign(**kwargs)
    model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'eth_'] 
                                   for metric in ['Close','Volume','close_off_high','volatility']]]
    # need to reverse the data frame so that subsequent rows represent later timepoints
    model_data = model_data.sort_values(by='Date')
    model_data.head()
    return model_data

"""
API ENDPOINTS
"""

def index(request):
    message = ""
    return render(request, 'trend_bot/pages/index.html', { 'message': "Welcome ! Let's start the step 1." })
    # return HttpResponse("Done")

def step1(request):
    # eth = crypto('ethereum')
    # eth.market_info.columns = eth.market_info.columns.str.replace("*", "")
    # eth.market_info.head()
    from . import step1
    fig = step1.start(btc.img, btc.market_info)
    return renderImage(fig)

def step2(request):
    from . import step2
    market_info = prepare()
    # step2.start(btc.img, market_info)
    fig = step2.start(btc.img, btc.market_info)
    return renderImage(fig)

def step3(request):
    from . import step3
    model_data = prepareLtsm(prepare())
    step3.start(btc.img, eth.img, model_data, eth_model)
    message = ""
    return render(request, 'trend_bot/pages/index.html', { 'message': "Step 3 has finished you can run step 4." })

def step4(request):
    # from .steps import step1
    message = ""
    return render(request, 'trend_bot/pages/index.html', { 'message': "Step 4 has finished." })

def step5(request):
    # from .steps import step1
    message = ""
    return render(request, 'trend_bot/pages/index.html', { message: "Step 5 has finished." })

def renderImage(fig):
    import io
    # Code that sets up figure goes here; in the question, that's ...
    FigureCanvasAgg(fig)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response
    return response 