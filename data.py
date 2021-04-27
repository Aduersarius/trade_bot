from numpy.lib.npyio import loads
from twelvedata import TDClient
apikey = "7b0d2ab7a04041f3999b57dcc508867c"
td = TDClient(apikey)
import time
from tqdm import tqdm
from random import randrange
import datetime
from datetime import date
import numpy as np
import copy
import pandas as pd


def get_random_time(whole_day=True, crypto=True):
    start = datetime.datetime(2020, 5, 1, 16, 00)
    current = start
    diff = (date.today().day - start.day) + (date.today().month - start.month) * 30 + (
                date.today().year - start.year) * 365
    current = current + datetime.timedelta(days=randrange(diff))
    current = current + datetime.timedelta(minutes=randrange(380))
    if whole_day and not crypto:
        current = datetime.datetime(2020, 4, 1, 16, 00)
        current = current + datetime.timedelta(days=randrange(diff))
    if current.isoweekday() in [6, 7] and not crypto:
        return get_random_time(whole_day)
    return current





def get_data(ticker="BTC/USD", interval="1min", window=1440, pred_window=0, size=5, whole_day=True, crypto=True):
    x, y = [], []
    r = [i for i in range(size)]
    for _ in tqdm(r):
        try:
            td = TDClient(apikey)
            df = td.time_series(
                symbol=ticker,
                outputsize=window+pred_window,
                interval=interval,
                timezone='America/New_York',
                start_date='2020-01-01 9:30',
                end_date=get_random_time(whole_day,crypto).strftime("20%y-%m-%d %H:%M")
            ).with_avgprice().without_ohlc()

            df = df.with_rsi().with_macd().with_ultosc().with_kst().with_supertrend(multiplier= 7,
         period= 2).with_adx().with_tsf()

            df = df.as_pandas().iloc[::-1]

        except Exception as e:
            time.sleep(1)
            r.append(0)
            print(e)
            continue

        if size == 1:
            return np.array(df).astype(np.float16)

        x.append(df.iloc[:window])
        y.append(df.iloc[window:])
    assert len(x) == size
    return x,y


def get_month(ticker="BTC/USD", interval="1min", window=1440, pred_window=0, size=30, whole_day=True, crypto=True):
    x, y = [], []
    df_ = pd.DataFrame()
    t = get_random_time(whole_day,crypto)
    r = [i for i in range(size)]
    for i in r:
        try:
            td = TDClient(apikey)
            df = td.time_series(
                symbol=ticker,
                outputsize=window+pred_window,
                interval=interval,
                timezone='America/New_York',
                start_date='2015-01-01 9:30',
                end_date=t+datetime.timedelta(days=i)
            ).with_avgprice().without_ohlc()

        #     df = df.with_rsi().with_macd().with_ultosc().with_kst().with_supertrend(multiplier= 7,
        #  period= 2).with_adx().with_tsf()
            df = df.with_rsi().with_macd().with_ultosc().with_kst().with_supertrend(multiplier= 7,
                period= 2).with_adx().with_tsf()
            df = df.with_supertrend(multiplier=7, period=2
            ).with_supertrend(multiplier=7, period=10
            ).with_supertrend(multiplier=6, period=2
            ).with_supertrend(multiplier=6, period=10
            ).with_supertrend(multiplier=5, period=2
            ).with_supertrend(multiplier=5, period=10
            ).with_supertrend(multiplier=4, period=2
            ).with_supertrend(multiplier=4, period=10
            ).with_supertrend(multiplier=3, period=2
            ).with_supertrend(multiplier=3, period=30
            ).with_supertrend(multiplier=2, period=2
            ).with_supertrend(multiplier=2, period=10
            ).with_supertrend(multiplier=10, period=2
            ).with_supertrend(multiplier=10, period=10
            ).with_supertrend(multiplier=14, period=2
            ).with_supertrend(multiplier=14, period=20
            ).with_supertrend(multiplier=7, period=30)

            df = df.as_pandas().iloc[::-1]
            df_ = pd.concat([df_, df], join="outer")
        
        except Exception as e:
            time.sleep(1)
            r.append(0)
            print(e)
            continue

    if len(df_) != window*size:
        print(len(df_))
        return get_month(ticker, window=1440, size=30)
    #assert len(x) == size
    return df_


def download_data():
    data, t = [], []
    crypto = ["DOGE/USD", "XRP/USD", "ETH/USD", "BTC/USD", "LTC/USD", "LINK/USD", "ADA/USD", "XLM/USD", "XMR/USD"]
    stocks = ["AAPL","SPCE","ZYNE","MSFT","FB","RIG","NFLX","NVDA","BYND"]
    t = tqdm(total=90)
    for stock in stocks:
        for _ in range(10):
            x = get_month(stock, window=390, size=30, crypto=False)
            x = np.array(x).astype(np.float16)
            data.append(x)
            t.update()
    data = np.array(data).astype(np.float16)
    np.save(file="C:\\Users\\emire\\vs_repositories\\strategies\\data\\stocks", arr=data)
    return data

if __name__ == "__main__":
    # x, y = get_data()
    # print(len(x))
    # print(x[0])
    # print(x[0].shape)
     data = download_data()
     print(len(data))