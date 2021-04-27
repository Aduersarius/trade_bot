import copy
import gym
from gym import spaces
import numpy as np
from data import get_data
from stable_baselines3.common.env_checker import check_env
from torch.utils.tensorboard import SummaryWriter
import pathlib, os
from datetime import datetime

class Environment(gym.Env):
    def __init__(self, window_length=30, is_eval=False, writer=0, discrete=False):
        super(Environment, self).__init__()
        self.is_eval = is_eval
        self.window_length = window_length
        self.writer = writer
        self.episode = 0
        self.reset()
        self.discrete = discrete
        if self.discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float16) 
        self.observation_space = spaces.Box(low=-255, high=255, shape=self.data[:self.window_length].flatten().shape, dtype=np.float16)
        self.eval_profit, self.eval_actions, self.eval_diff = [],[],[]
        
        

    def reset(self):
        self.data = get_data(ticker="BTC/USD", window=1440, size=1)
        self.episode += 1
        self.t = 0
        self.done = False
        self.profit = 1
        self.action_b = 0
        self.action_s = 0
        self.reward = 0
        self.cash = 100000
        self.balance = 100000
        self.shares = 0
        self.bought = 0
        self.money_flow = 0
        try:
            self.window = copy.deepcopy(self.data[:self.window_length])
            self.window[:,0] /= self.window[0,0]
        except BaseException as e:
            print(e)
            self.reset()
        return self.window.flatten()

    def step(self, action):
        if self.discrete:
            if action == 0:  # buy
                if self.bought == 0:
                    self.bought = self.data[self.t+self.window_length-1,0]
                    self.action_b += 1
                    self.shares += self.cash / self.bought
                    self.cash -= self.shares * self.bought
            elif action == 1:  # sell
                if self.bought != 0:
                    self.action_s += 1
                    self.cash += self.shares * self.data[self.t+self.window_length-1,0]
                    self.shares = 0
                    self.profit *= self.data[self.t+self.window_length-1, 0] / self.bought
                    self.bought = 0
            elif action == 2:  # hold
                pass
        else:
            if action[0] > 0 and self.cash > 0:
                amount = action[0]*self.cash
                self.money_flow += amount
                self.cash -= amount
                self.shares += amount/self.data[self.t+self.window_length-1,0]
            if action[0] < 0 and self.shares > 0:
                amount = -1*action[0]*self.shares
                self.cash += amount*self.data[self.t+self.window_length-1,0]
                self.money_flow += amount*self.data[self.t+self.window_length-1,0]
                self.shares -= amount

        
        reward = (((self.cash + self.shares * self.data[self.t+self.window_length-1, 0]) / self.balance)-1)*10000
        self.t += 1
        self.balance = self.cash + self.shares * self.data[self.t+self.window_length-1, 0]
        #reward = (self.balance/100000 - self.buy_hold() - 0.03)*100

        self.window = copy.deepcopy(self.data[self.t:self.t+self.window_length])
        self.window[:,0] /= self.window[0,0]
        self.info = {"reward": self.reward,
                "money flow": self.money_flow/100000,
                "balance_ratio": self.balance/100000,
                "buy&hold": self.buy_hold(),
                "max profit" : self.max_rew()}

        if self.t+self.window_length == len(self.data)-1 and not self.is_eval:
            self.done = True
            print(self.info)
            self.writer.add_scalar(tag="Reward/episode", scalar_value=(self.info['reward']-1)*100, global_step=self.episode)
            self.writer.add_scalar(tag="Money flow/episode", scalar_value=self.info['money flow'], global_step=self.episode)
            self.writer.add_scalar(tag="Diff/episode", scalar_value=(self.info['balance_ratio']-self.info['buy&hold'])*100, global_step=self.episode)
            self.writer.flush()
        
        if self.t+self.window_length == len(self.data)-1 and self.is_eval:
            self.done = True
            self.eval_profit.append((self.info['balance_ratio']-1)*100)
            self.eval_actions.append(self.info['money flow'])
            self.eval_diff.append((self.info['balance_ratio']-self.info['buy&hold'])*100)
            if self.episode % 20 == 0:
                print("=======================================EVALUATION============================================")
                print(f"|| Profits: {'{:.2%}'.format(np.mean(self.eval_profit))} || Actions: {np.mean(self.eval_actions)} || Diff: {'{:.2%}'.format(np.mean(self.eval_diff)/100)} ||")
                print("=============================================================================================")
                self.writer.add_scalar(tag="Profit_eval/episode", scalar_value=np.mean(self.eval_profit), global_step=self.episode/20)
                self.writer.add_scalar(tag="Moneyflow_eval/episode", scalar_value=np.mean(self.eval_actions), global_step=self.episode/20)
                self.writer.add_scalar(tag="Diff_eval/episode", scalar_value=np.mean(self.eval_diff), global_step=self.episode/20)
                self.writer.flush()
        
        
        self.reward += reward
        return self.window.flatten(), reward, self.done, self.info


    def buy_hold(self):
        b = np.float(1.0)
        for i in range(self.window_length, self.window_length+self.t):
            b *= self.data[i, 0]
        return b

    def max_rew(self):
        r = np.float(1.0)
        for i in range(self.window_length, self.window_length+self.t):
            if self.data[i, 0] > 1:
                r *= self.data[i, 0]
        return r


if __name__ == "__main__":
    print(check_env(Environment()))

