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


def get_random_time(whole_day=True, crypto=True):
    start = datetime.datetime(2020, 6, 1, 16, 00)
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
    return current.strftime("20%y-%m-%d %H:%M")


def get_data(ticker="AAPL", interval="1min", window=10, size=2, whole_day=True):
    x, y = [], []
    for _ in range(size):
        try:
            td = TDClient(apikey)
            df = td.time_series(
                symbol=ticker,
                outputsize=window,
                interval=interval,
                timezone='America/New_York',
                start_date='2020-01-01 9:30',
                end_date=get_random_time(whole_day)
            ).with_avgprice()

            df = df.with_rocr(time_period=1).with_rsi().with_macd().with_ultosc()

            df = df.as_pandas().iloc[::-1].to_numpy().astype(np.float16)

        except Exception as e:
            time.sleep(1)
            size += 1
            print(e)
            continue
        

    return np.array(df)

if __name__ == "__main__":
    print(get_data())
#
#print(x)
#print(get_random_time())

