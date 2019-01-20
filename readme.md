# trend-bot

***This is a personal test to know more about machine learning 
[WIP]***

Based on this [tutotial](https://activewizards.com/blog/bitcoin-price-forecasting-with-deep-learning-algorithms/)

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
2. Rename the file to data.txt
3. Put it at project root

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