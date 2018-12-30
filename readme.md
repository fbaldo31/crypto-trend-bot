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

**1.Create a python3.6 environnment**

**Add a template file:**
>`sudo mv ~/trend_bot/learning_bot/pages/index.html ~/tools/anaconda3/envs/py36/lib/python3.7/site-packages/django/contrib/admin/templates/trend_bot/pages/index.html
`

## Run

`anaconda-navigator`

1. Activate your environment:
Assuming your env is `py36` `source activate py36`
2. `cd trend_bot` 
3. `python manage.py runserver 8000 --nothreading --noreload`

## Features

1. Get the BTC (can be either other crypto) trend over the past year.
2. Get the BTC rates for each min during the last week
3. Put the high, mid and low values in arrays and preprocess it.
4. Fetch every minute the current rate.
