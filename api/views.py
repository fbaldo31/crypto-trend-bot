# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
import json
# Create your views here.
from django.http import HttpResponse
from django.template import Template, Context

import matplotlib.pyplot as plt
from . import ltms

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
    return HttpResponse("Welcome")

def intro(request):
    return ltms.start('file')

def step1(request):
    body = json.dumps(ltms.get_seasonal_decompose())
    return HttpResponse(body, content_type='application/json')

def step2(request):
    body = json.dumps(ltms.get_training_process())
    return HttpResponse(body, content_type='application/json')

def autocorrelation():
    body = json.dumps(ltms.get_autocorrelation())
    return HttpResponse(body, content_type='application/json')

def step3(request):
    body = json.dumps(ltms.get_results_demonstrating0())
    return HttpResponse(body, content_type='application/json')

def step4(request):
    body = json.dumps(ltms.get_results_demonstrating1())
    return HttpResponse(body, content_type='application/json')

# WIP
def prevision(request):
    from . import pipeline
    train1 = request.GET['1start']
    train2 = request.GET['1end']
    test1 = request.GET['2start']
    test2 = request.GET['2end']
    body = json.dumps(pipeline.runPipeline(train1, train2, test1, test2))
    return HttpResponse(body, content_type='application/json')