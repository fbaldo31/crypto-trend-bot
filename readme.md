# trend-bot

***This is a personal test to know more about machine learning 
[WIP] Not usefull ATM***

## Pre-requisites

- Python 3.7
- anaconda 3
- slicklearn and all its required dependencies
- matplotlib
- django
- seaborn
- pil
- lxml

## Settings

Replace <YOUR_API_KEY> in `learning_bot/crypto_bot.py` with your own, got on [cryptocompare](https://min-api.cryptocompare.com)

## Run

`anaconda-navigator`

then open your environment in a terminal and

`python manage.py runserver 8000 --nothreading --noreload`

## Features

1. Get the BTC (can be either other crypto) trend over the past year.
2. Get the BTC rates for each min during the last week
3. Put the high, mid and low values in arrays and preprocess it.
4. Fetch every minute the current rate.
