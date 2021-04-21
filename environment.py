import copy
import gym
from gym import spaces
import numpy as np

class Environment(gym.Env):
    def __init__(self, data, window_length=64,):
        super(Environment, self).__init__()
        self.data = data
        self.window_length = window_length
        print(self.data[:window_length].shape,"ofuewh")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.data[:window_length].shape, dtype=np.float16)
        
        self.reset()
        

    def reset(self):
        self.t = 0
        self.done = False
        self.profit = 1
        self.action_b = 0
        self.action_s = 0
        self.cash = 100000
        self.balance = self.cash
        self.shares = 0
        self.bought = 0
        self.window = copy.deepcopy(self.data[:self.window_length])
        self.window[:, 0] = (self.window[:, 0] / self.window[0, 0]) - 1
        return self.window

    # general learning function
    def step(self, action):

        if action == 0:  # buy
            if self.bought == 0:
                self.bought = self.data.iloc[self.t+self.window_length-1]['rocr']
                self.action_b += 1
                self.shares += self.cash / self.bought
                self.cash -= self.shares * self.bought
        elif action == 1:  # sell
            if self.bought != 0:
                self.action_s += 1
                self.cash += self.shares * self.data.iloc[self.t+self.window_length-1]['rocr']
                self.shares = 0
                self.profit *= self.data.iloc[self.t+self.window_length-1]['rocr'] / self.bought
                self.bought = 0
        elif action == 2:  # hold
            pass
        else:
            print(action)

        self.t += 1
        reward = (((self.cash + self.shares * self.data.iloc[self.t+self.window_length-1, :]['rocr']) / self.balance)-1)*10
        self.balance = self.cash + self.shares * self.data.iloc[self.t+self.window_length-1, :]['rocr']

        self.window = copy.deepcopy(self.data.iloc[self.t:self.t+self.window_length].to_numpy())
        self.window[:, 0] = self.window[:, 0] / self.window[0, 0] - 1

        stats = [reward, self.profit, self.action_b + self.action_s, self.balance/100000,
                 self.data.iloc[-1, :]['rocr']/self.data.iloc[0, :]['rocr']]

        if self.t+self.window_length == len(self.data):
            self.done = True
        print(self.window.shape,"hfdos")
        return self.window, reward, self.done, stats

    # test rsi strategy
    def rsi(self, rsi):
        if self.data.iloc[self.t, :]["rsi"] < 40 - rsi and self.bought == 0:
            self.bought = self.data.iloc[self.t, :]['rocr']
            self.action_b += 1
            self.shares += self.cash / self.data.iloc[self.t, :]['rocr']
            self.cash -= self.shares * self.data.iloc[self.t, :]['rocr']

        elif self.data.iloc[self.t, :]["rsi"] > 60 + rsi and self.bought != 0:
            self.action_s += 1
            self.cash += self.shares * self.data.iloc[self.t, :]['rocr']
            self.shares = 0
            self.profit *= (self.data.iloc[self.t, :]['rocr']) / self.bought
            self.bought = 0

        #self.balance = self.cash + self.shares * self.data.iloc[self.t + self.window_length - 1, :]['rocr']
        # print(self.data.iloc[0, :])
        # print(self.data.iloc[1, :])
        stats = [0, self.profit, self.action_b + self.action_s, (self.data.iloc[self.t, :]['rocr']/self.data.iloc[0, :]['rocr']) - 1,
                 ((self.cash + self.shares * self.data.iloc[self.t, :]['rocr']) / 100000) - 1]
        self.t += 1
        return stats

    # test supertrend strategy
    def supertrend(self, action=0):
        uptrend = 0 if self.data.iloc[self.t]["rocr"] / self.data.iloc[self.t]["supertrend"] < 1 else 1
        if self.uptrend != uptrend and not self.uptrend and self.bought == 0:
            self.bought = self.data.iloc[self.t, :]['rocr']
            self.action_b += 1
            self.shares += self.cash / self.data.iloc[self.t, :]['rocr']
            self.cash -= self.shares * self.data.iloc[self.t, :]['rocr']

        elif self.uptrend != uptrend and not self.uptrend and self.bought != 0:
            self.action_s += 1
            self.cash += self.shares * self.data.iloc[self.t, :]['rocr']
            self.shares = 0
            self.profit *= (self.data.iloc[self.t, :]['rocr']) / self.bought
            self.bought = 0

        self.uptrend = uptrend
        self.t += 1
        stats = [0, self.profit, self.action_b + self.action_s, self.data.iloc[self.t, :]['rocr'] - 1,
                 ((self.cash + self.shares * self.data.iloc[self.t, :]['rocr']) / 1000) - 1]

        return stats

