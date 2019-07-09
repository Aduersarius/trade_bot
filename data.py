from twelvedata import TDClient
apikey = "7b0d2ab7a04041f3999b57dcc508867c"
td = TDClient(apikey)
import time
from tqdm import tqdm
from random import randrange
import datetime
from datetime import date


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


def get_data(ticker="AAPL", interval="1min", feed_window=10, prediction_window=1, size=30, whole_day=True):
    x, y = [], []
    for _ in tqdm(range(size)):
        try:
            df = td.time_series(
                symbol=ticker,
                outputsize=feed_window + prediction_window,
                interval=interval,
                timezone='America/New_York',
                start_date='2020-01-01 9:30',
                end_date=get_random_time(whole_day)
            )

            df = df.with_rsi().with_supertrend()

            df = df.as_pandas().drop(['high', 'low', 'open'], axis=1).iloc[::-1]
            #df = df.to_numpy()
            #df = np.array(df)
        except Exception as e:
            time.sleep(1)
            size += 1
            print(e)
            continue
        x.append(df.iloc[:feed_window])
        y.append(df.iloc[feed_window:])

    return x, y


#x, y = get_data(ticker="BTC/USD", feed_window=1440, prediction_window=0, size=3)
#print(x)

