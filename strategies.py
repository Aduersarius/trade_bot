from modules.environment import *
from data import get_data
import copy
import numpy as np


def run(env):
    stats = []
    for step in range(len(env.data) - 1):
        stats = env.rsi(15)

    return stats


def main():
    size = 5
    profit, profit_b, diff_, a = [], [], [], []
    for i in range(0, 20, 2):
        profits = []
        diff = []
        profits2, act = [], []
        tickers = ["XRP/USD", "ETH/USD", "BTC/USD", "LTC/USD", "LINK/USD", "ADA/USD", "XLM/USD", "XMR/USD"]
        for ticker in tickers:
            data, _ = get_data(ticker=ticker, feed_window=1440, prediction_window=0, size=size)
            for episode in range(size):
                # try:
                    stats = run(Environment(data=copy.deepcopy(data[episode])))

                    print("Profit rsi:", "{:.2%}".format(stats[1] - 1), "  Actions:", stats[2])
                    print("Buy&hold:", "{:.2%}".format(stats[3]), "  diff:",
                          "{:.2%}".format(stats[1] - stats[3] - 1), " Profits balance:", "{:.2%}".format(stats[4]))
                    print("------------------------------------------------------------------------------------")
                    profits.append(stats[1] - 1)
                    diff.append(stats[1] - stats[3] - 1)
                    profits2.append(stats[4])
                    act.append(stats[2])

                # except Exception as e:
                #     print(e)

        print("------------------------------------------------------------------------------------")
        print("---------------- Profit:", "{:.2%}".format(np.mean(profits)), "-------------------")
        print("---------------- Profit with balance:", "{:.2%}".format(np.mean(profits2)), "-------------------")
        print("----------------  Diff:", "{:.2%}".format(np.mean(diff)), "-------------------")
        print("------------------------------------------------------------------------------------")
        profit.append(np.mean(profits))
        profit_b.append(np.mean(profits2))
        diff_.append(np.mean(diff))
        a.append(np.mean(act))

    for i in range(len(profit)):
        print("========================================================================")
        print(40 - 2 * i, "и", 60 + 2 * i, "|| profit:", "{:.2%}".format(profit[i]), "|| profit balance:",
              "{:.2%}".format(profit_b[i]), "|| diff:", "{:.2%}".format(diff_[i]), "|| actions:", a[i])


if __name__ == "__main__":

    _ = main()
