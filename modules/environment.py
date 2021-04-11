import copy

class Environment:
    def __init__(self, data, window_length=64,):

        self.data = data
        self.window_length = window_length
        self.reset()
        self.uptrend = 0 if data.iloc[0]["close"] / data.iloc[0]["supertrend"] < 1 else 1  # for supertrend

    # reset after each epoch
    def reset(self):
        self.t = 0
        self.profit = 1
        self.action_b = 0
        self.action_s = 0
        self.cash = 100000
        self.balance = self.cash
        self.shares = 0
        self.bought = 0
        self.window = copy.deepcopy(self.data.iloc[:self.window_length].to_numpy())
        self.window[:, 0] = (self.window[:, 0] / self.window[0, 0]) - 1

        return self.window

    # general learning function
    def __call__(self, action):

        if action == 0:  # buy
            if self.bought == 0:
                self.bought = self.data.iloc[self.t+self.window_length-1]['close']
                self.action_b += 1
                self.shares += self.cash / self.bought
                self.cash -= self.shares * self.bought
        elif action == 1:  # sell
            if self.bought != 0:
                self.action_s += 1
                self.cash += self.shares * self.data.iloc[self.t+self.window_length-1]['close']
                self.shares = 0
                self.profit *= self.data.iloc[self.t+self.window_length-1]['close'] / self.bought
                self.bought = 0
        elif action == 2:  # hold
            pass
        else:
            print(action)

        self.t += 1
        reward = (((self.cash + self.shares * self.data.iloc[self.t+self.window_length-1, :]['close']) / self.balance)-1)*10
        self.balance = self.cash + self.shares * self.data.iloc[self.t+self.window_length-1, :]['close']

        self.window = copy.deepcopy(self.data.iloc[self.t:self.t+self.window_length].to_numpy())
        self.window[:, 0] = self.window[:, 0] / self.window[0, 0] - 1

        stats = [reward, self.profit, self.action_b + self.action_s, self.balance/100000,
                 self.data.iloc[-1, :]['close']/self.data.iloc[0, :]['close']]

        return self.window, stats

    # test rsi strategy
    def rsi(self, rsi):
        if self.data.iloc[self.t, :]["rsi"] < 40 - rsi and self.bought == 0:
            self.bought = self.data.iloc[self.t, :]['close']
            self.action_b += 1
            self.shares += self.cash / self.data.iloc[self.t, :]['close']
            self.cash -= self.shares * self.data.iloc[self.t, :]['close']

        elif self.data.iloc[self.t, :]["rsi"] > 60 + rsi and self.bought != 0:
            self.action_s += 1
            self.cash += self.shares * self.data.iloc[self.t, :]['close']
            self.shares = 0
            self.profit *= (self.data.iloc[self.t, :]['close']) / self.bought
            self.bought = 0

        #self.balance = self.cash + self.shares * self.data.iloc[self.t + self.window_length - 1, :]['close']
        # print(self.data.iloc[0, :])
        # print(self.data.iloc[1, :])
        stats = [0, self.profit, self.action_b + self.action_s, (self.data.iloc[self.t, :]['close']/self.data.iloc[0, :]['close']) - 1,
                 ((self.cash + self.shares * self.data.iloc[self.t, :]['close']) / 100000) - 1]
        self.t += 1
        return stats

    # test supertrend strategy
    def supertrend(self, action=0):
        uptrend = 0 if self.data.iloc[self.t]["close"] / self.data.iloc[self.t]["supertrend"] < 1 else 1
        if self.uptrend != uptrend and not self.uptrend and self.bought == 0:
            self.bought = self.data.iloc[self.t, :]['close']
            self.action_b += 1
            self.shares += self.cash / self.data.iloc[self.t, :]['close']
            self.cash -= self.shares * self.data.iloc[self.t, :]['close']

        elif self.uptrend != uptrend and not self.uptrend and self.bought != 0:
            self.action_s += 1
            self.cash += self.shares * self.data.iloc[self.t, :]['close']
            self.shares = 0
            self.profit *= (self.data.iloc[self.t, :]['close']) / self.bought
            self.bought = 0

        self.uptrend = uptrend
        self.t += 1
        stats = [0, self.profit, self.action_b + self.action_s, self.data.iloc[self.t, :]['close'] - 1,
                 ((self.cash + self.shares * self.data.iloc[self.t, :]['close']) / 1000) - 1]

        return stats

