class Environment:
    def __init__(self, data, window_length=50, rsi=0):
        self.data = data
        self.init_price = data.iloc[0]["close"]
        self.window_length = window_length
        self.reset()
        self.rsi_ = rsi  # for rsi
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
        #self.history = [0 for _ in range(self.history_length)]
        self.window = self.data.iloc[:self.window_length].to_numpy()
        self.window[:, 0] = (self.window[:, 0] / self.window.iloc[0, 0]) - 1
        # return the state/observation representation
        return self.window

    # general learning function
    def __call__(self, action):
        if action == 1 and self.bought == 0:  # buy
            self.bought = self.data.iloc[self.t]['close']
            self.action_b += 1
            self.shares += self.cash / (self.init_price * self.bought)
            self.cash -= self.shares * (self.init_price * self.bought)
        elif action == 2 and self.bought != 0:  # sell
            self.action_s += 1
            self.cash += self.shares * self.init_price * self.data.iloc[self.t]['close']
            self.shares = 0
            self.profit *= (self.data.iloc[self.t]['close']) / self.bought
            self.bought = 0
        elif action == 3:  # hold
            pass
        else:
            raise ValueError("invalid action")

        reward = (((self.cash + self.shares * self.init_price * self.data.iloc[self.t, :]['close']) / self.balance)-1)*10
        self.balance = self.cash + self.shares * self.init_price * self.data.iloc[self.t, :]['close']

        self.window = self.data.iloc[:self.window_length].to_numpy()
        self.window[:, 0] = self.window[:, 0] / self.window.iloc[0, 0] - 1

        self.window[:-1] = self.window[1:]
        self.window[-1] = self.data.iloc[self.t, :]
        stats = [reward, self.profit, self.action_b + self.action_s, self.data.iloc[len(self.data), :]['close']]
        self.t += 1
        return self.window, stats

    # test rsi strategy
    def rsi(self, action=0):
        if self.data.iloc[self.t, :]["rsi"] < 40 - self.rsi_ and self.bought == 0:
            self.bought = self.data.iloc[self.t, :]['close']
            self.action_b += 1
            self.shares += self.cash / (self.init_price * self.data.iloc[self.t, :]['close'])
            self.cash -= self.shares * (self.init_price * self.data.iloc[self.t, :]['close'])

        elif self.data.iloc[self.t, :]["rsi"] > 60 + self.rsi_ and self.bought != 0:
            self.action_s += 1
            self.cash += self.shares * (self.init_price * self.data.iloc[self.t, :]['close'])
            self.shares = 0
            self.profit *= (self.data.iloc[self.t, :]['close']) / self.bought
            self.bought = 0

        self.t += 1
        stats = [0, self.profit, self.action_b + self.action_s, self.data.iloc[self.t, :]['close'] - 1,
                 ((self.cash + self.shares * self.init_price * self.data.iloc[self.t, :]['close']) / 1000) - 1]

        return stats

    # test supertrend strategy
    def supertrend(self, action=0):
        uptrend = 0 if self.data.iloc[self.t]["close"] / self.data.iloc[self.t]["supertrend"] < 1 else 1
        if self.uptrend != uptrend and not self.uptrend and self.bought == 0:
            self.bought = self.data.iloc[self.t, :]['close']
            self.action_b += 1
            self.shares += self.cash / (self.init_price * self.data.iloc[self.t, :]['close'])
            self.cash -= self.shares * (self.init_price * self.data.iloc[self.t, :]['close'])

        elif self.uptrend != uptrend and not self.uptrend and self.bought != 0:
            self.action_s += 1
            self.cash += self.shares * (self.init_price * self.data.iloc[self.t, :]['close'])
            self.shares = 0
            self.profit *= (self.data.iloc[self.t, :]['close']) / self.bought
            self.bought = 0

        self.uptrend = uptrend
        self.t += 1
        stats = [0, self.profit, self.action_b + self.action_s, self.data.iloc[self.t, :]['close'] - 1,
                 ((self.cash + self.shares * self.init_price * self.data.iloc[self.t, :]['close']) / 1000) - 1]

        return stats

