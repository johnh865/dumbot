# -*- coding: utf-8 -*-

from backtester import gym


env = gym.env_noise(0)

backtest = env.backtest
obs = env.reset()

colnames = next(iter(env.indicators.dataframes.values())).columns
my_ind = "indicator1()['diff(100)']"
my_index = list(colnames).index(my_ind)

print( env.step(1)[0][my_index])