import copy
import gym
from gym import spaces
import numpy as np

class Environment(gym.Env):
    def __init__(self, data, rsi):
        super(Environment, self).__init__()
        self.data = data
        self.rsi = rsi
        self.reset()
        

    def reset(self):
        self.t = 0
        self.done = False
        self.profit = 1
        self.action_b = 0
        self.action_s = 0
        self.cash = 100000
        self.balance = 100000
        self.shares = 0
        self.bought = 0
        # self.window = copy.deepcopy(self.data[:self.window_length])
        # self.window[:, 0] = (self.window[:, 0] / self.window[0, 0]) - 1
        return

    def buy(self):
        self.bought = self.data[self.t, 0]
        self.action_b += 1
        self.shares += self.cash / self.data[self.t, 0]
        self.cash -= self.shares * self.data[self.t, 0]
    
    def sell(self):
        self.action_s += 1
        self.cash += self.shares * self.data[self.t, 0]
        self.shares = 0
        self.profit *= (self.data[self.t, 0]) / self.bought
        self.bought = 0

    # general learning function
    def step(self, action):

        if action == 0:  # buy
            if self.bought == 0:
                self.bought = self.data.iloc[self.t+self.window_length-1]['avgprice']
                self.action_b += 1
                self.shares += self.cash / self.bought
                self.cash -= self.shares * self.bought
        elif action == 1:  # sell
            if self.bought != 0:
                self.action_s += 1
                self.cash += self.shares * self.data.iloc[self.t+self.window_length-1]['avgprice']
                self.shares = 0
                self.profit *= self.data.iloc[self.t+self.window_length-1]['avgprice'] / self.bought
                self.bought = 0
        elif action == 2:  # hold
            pass
        else:
            print(action)

        self.t += 1
        reward = (((self.cash + self.shares * self.data.iloc[self.t+self.window_length-1, :]['avgprice']) / self.balance)-1)*10
        self.balance = self.cash + self.shares * self.data.iloc[self.t+self.window_length-1, :]['avgprice']

        self.window = copy.deepcopy(self.data.iloc[self.t:self.t+self.window_length].to_numpy())
        self.window[:, 0] = self.window[:, 0] / self.window[0, 0] - 1

        stats = [reward, self.profit, self.action_b + self.action_s, self.balance/100000,
                 self.data.iloc[-1, :]['avgprice']/self.data.iloc[0, :]['avgprice']]

        if self.t+self.window_length == len(self.data):
            self.done = True
        print(self.window.shape,"hfdos")
        return self.window, reward, self.done, stats

    # test rsi strategy
    def _rsi(self, rsi=0):
        if (self.data[self.t, 1] < 40 - self.rsi) and self.bought == 0:
            self.buy()

        elif (self.data[self.t, 1] > 60 + self.rsi) and self.bought != 0: #and (self.bought < self.data[self.t, 0]):
            self.sell()

        #self.balance = self.cash + self.shares * self.data[self.t + self.window_length - 1, :]['avgprice']
        # print(self.data.iloc[0, :])
        # print(self.data.iloc[1, :])
        stats = [0, self.profit-1, self.action_b + self.action_s, (self.data[self.t, 0]/self.data[0, 0]) - 1,
                 ((self.cash + self.shares * self.data[self.t, 0]) / 100000) - 1]
        self.t += 1
        return stats

    # test supertrend strategy
    def supertrend(self, action=0):
        uptrend = 0 if self.data.iloc[self.t,0] / self.data.iloc[self.t]["supertrend"] < 1 else 1
        if self.uptrend != uptrend and not self.uptrend and self.bought == 0:
            self.bought = self.data.iloc[self.t, 0]
            self.action_b += 1
            self.shares += self.cash / self.data.iloc[self.t, 0]
            self.cash -= self.shares * self.data.iloc[self.t, 0]

        elif self.uptrend != uptrend and not self.uptrend and self.bought != 0:
            self.action_s += 1
            self.cash += self.shares * self.data.iloc[self.t, 0]
            self.shares = 0
            self.profit *= (self.data.iloc[self.t, 0]) / self.bought
            self.bought = 0

        self.uptrend = uptrend
        self.t += 1
        stats = [0, self.profit, self.action_b + self.action_s, self.data.iloc[self.t, 0] - 1,
                 ((self.cash + self.shares * self.data.iloc[self.t, 0]) / 1000) - 1]

        return stats

