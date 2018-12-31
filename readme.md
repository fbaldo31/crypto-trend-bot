# trend-bot

***This is a personal test to know more about machine learning 
[WIP] Not usefull ATM***

Based on this [tutotial](https://activewizards.com/blog/bitcoin-price-forecasting-with-deep-learning-algorithms/)

## Pre-requisites

**Backend**
- Python 3.7
- anaconda 3
- slicklearn and all its required dependencies
- matplotlib
- django
- seaborn
- pil
- lxml
- keras
- plotly

**Frontend**
- Node.js
- @angular/cli (global)

## Settings

Replace <YOUR_API_KEY> in `api/ltms.py` with your own, got on [cryptocompare](https://min-api.cryptocompare.com)

**1.Create a python3.6 environnment**


## Run

**Backend**

1. Activate your Python environment:
Assuming your env is `py36` `source activate py36`
2. Open Anaconda Navigator to install the required dependencies `anaconda-navigator`
3. Run server `python3 manage.py runserver 8000 --nothreading --noreload`

**Frontend**
`cd front`

`ng serve`

## Features

1. Get the BTC (can be either other crypto) trend over the past year.
2. Train the model
3. Display results in charts
