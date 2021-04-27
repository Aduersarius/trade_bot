from environment import *
from data import get_data
import copy
import numpy as np
from tqdm import tqdm


def run(env):
    stats = []
    for step in range(len(env.data) - 1):
        stats = env._rsi()

    return stats


def main():
    size = 100
    worst_month, profit, profit_b, diff_, a = [], [], [], [], []
    
    for i in tqdm(range(5, 25, 1)):
        ll_profit, ll_balance = 1000, 1000
        profits = []
        diff = []
        profits2, act = [], []
        tickers = ["XRP/USD", "ETH/USD", "BTC/USD", "LTC/USD", "LINK/USD", "ADA/USD", "XLM/USD", "XMR/USD"]
        d = np.load("C:\\Users\\emire\\vs_repositories\\trade_bot\\data\\crypto_month.npy")
        #d=d.reshape(800,1440,6)
        # for ticker in tickers:
        #     data, _ = get_data(ticker=ticker, window=1440, pred_window=0, size=size)
        for episode in d:
            #try:
                stats = run(Environment(data=copy.deepcopy(episode), rsi=i))

                # print("Profit rsi:", "{:.2%}".format(stats[1] - 1), "  Actions:", stats[2])
                # print("Buy&hold:", "{:.2%}".format(stats[3]), "  diff:",
                #         "{:.2%}".format(stats[1] - stats[3] - 1), " Profits balance:", "{:.2%}".format(stats[4]))
                # print("------------------------------------------------------------------------------------")
                profits.append(stats[1])
                diff.append(stats[1] - stats[3])
                profits2.append(stats[4])
                act.append(stats[2])
                if stats[1] < ll_profit:
                    ll_profit = stats[1]
                if stats[4] < ll_balance:
                    ll_balance = stats[4]

            # except Exception as e:
            #     print(e)

        # print("------------------------------------------------------------------------------------")
        # print("---------------- Profit:", "{:.2%}".format(np.mean(profits)), "-------------------")
        # print("---------------- Profit with balance:", "{:.2%}".format(np.mean(profits2)), "-------------------")
        # print("----------------  Diff:", "{:.2%}".format(np.mean(diff)), "-------------------")
        # print("------------------------------------------------------------------------------------")
        profit.append(np.mean(profits))
        profit_b.append(np.mean(profits2))
        diff_.append(np.mean(diff))
        a.append(np.mean(act))
        worst_month.append([ll_profit, ll_balance])

    for i in range(len(profit)):
        print("========================================================================")
        print(40 - i, "и", 60 + i, "|| profit:", "{:.2%}".format(profit[i]), "|| profit balance:",
              "{:.2%}".format(profit_b[i]), "|| diff:", "{:.2%}".format(diff_[i]), "|| actions:", a[i], 
              f" || worst month: {'{:.2%}'.format(worst_month[i][0])} and {'{:.2%}'.format(worst_month[i][1])}")


if __name__ == "__main__":

    _ = main()
