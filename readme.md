# trend-bot

***This is a personal test to know more about machine learning [WIP]***

Based on this [tutotial](https://activewizards.com/blog/bitcoin-price-forecasting-with-deep-learning-algorithms/)

[![Codefresh build status]( https://g.codefresh.io/api/badges/pipeline/fbaldo31/fbaldo31%2Fcrypto-trend-bot%2Fcrypto-trend-bot?branch=master&key=eyJhbGciOiJIUzI1NiJ9.NWM2MTFjNmMxMmE5NTYyZTZhYWQ3YTRl.S6H0kL3RuTkulI3shIONacgTBojrmumEo9peBrC4buw&type=cf-1)]( https://g.codefresh.io/pipelines/crypto-trend-bot/builds?repoOwner=fbaldo31&repoName=crypto-trend-bot&serviceName=fbaldo31%2Fcrypto-trend-bot&filter=trigger:build~Build;branch:master;pipeline:5c611d41f8862d6b3ec95638~crypto-trend-bot)

## Pre-requisites

### Backend

You have 2 options :

- Run the server with docker
- Follow the indications:

1. Create a python3.5+ environment (can be done with anaconda3)
2. Install the following packages. Can be done with Anaconda Navigator: `anaconda-navigator`

- slicklearn
- matplotlib
- django
- seaborn
- pil
- lxml
- keras
- plotly

### Frontend

- Node.js
- @angular/cli (global)
- Yarn

## Settings

1. Get data [here](https://www.kaggle.com/mczielinski/bitcoin-historical-data/data)
or [here](https://www.kaggle.com/kognitron/zielaks-bitcoin-historical-data-wo-nan)

2. Rename the file to data.csv and replace the placeholder file at projet root

## Run

### Back

1. Activate your Python environment: Assuming your env is `py36` `source activate py36`
2. Run server `python3 manage.py runserver 8000 --nothreading --noreload`

### Front

1. Install depencencies `cd front && yarn`
2. Run `ng serve`

## Features

1. Get the BTC (can be either other crypto) trend over the past year.
2. Train the model
3. Display results in charts

## Run Server with Docker

1. Add the data file as decribed before
2. First build the image: `docker build . -t trend_bot`
3. Run the container `docker run --name trend_bot -p 8000:8000 trend_bot`

***note1:*** next time you can start it with `docker start trend_bot`

***note2:*** If the container freeze:

```bash
docker stop trend_bot
docker start trend_bot
docker exec -it trend_bot bash
python3 /manage.py runserver 0.0.0.0:8000 --nothreading --noreload
```

or Follow the pre-requisites to setup your environment.
