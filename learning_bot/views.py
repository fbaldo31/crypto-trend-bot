# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

# from . import bot
from . import crypto_bot

def index(request):
    # plt = "Hello Fred"
    # crypto_bot.start()
    return HttpResponse("Done")

    # return HttpResponse("Hello, world. You're at the trend_bot index.")