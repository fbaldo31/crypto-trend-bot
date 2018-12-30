# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
import json
# Create your views here.
from django.http import HttpResponse
from django.template import Template, Context

import matplotlib.pyplot as plt
from . import ltms
# class crypto():
#     def __init__(self, currency):
#         # get market info for bitcoin from the start of 2016 to the current day
#         self.market_info = pd.read_html("https://coinmarketcap.com/currencies/"+currency+"/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
#         # convert the date string to the correct date format
#         self.market_info.assign(Date=pd.to_datetime(self.market_info['Date']))

def renderImage(fig):
    import io
    # Code that sets up figure goes here; in the question, that's ...
    # FigureCanvasAgg(fig)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response

"""
API ENDPOINTS
"""

def index(request):
    message = ""
    # return render(request, 'trend_bot/pages/index.html', { 'message': "Welcome ! Let's start the step 1." })
    return HttpResponse("Welcome")

def step1(request):
    body = json.dumps(ltms.get_seasonal_decompose())
    return HttpResponse(body, content_type='application/json')