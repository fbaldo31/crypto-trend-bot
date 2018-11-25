# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.

class Currency:
    price: float
    volume_24h: float
    market_cap: float
    percent_change_1h: float 
    percent_change_24h: float
    percent_change_7d: float

class Quote:
    USD: Currency 
    EUR: Currency

class Crypto:
    id: int
    name: str
    symbol: str 
    website_slug: str
    rank: int
    circulating_supply: float
    total_supply: float
    max_supply: float
    quotes: Quote
    last_updated: int

class Metadata:
    timestamp: int 
    error: str

class Sequence:
    crypto: Crypto
    timestamp: int
    rate: int

    def __init__(self, data, **kwargs):
        self.crypto = data