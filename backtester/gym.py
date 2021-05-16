# -*- coding: utf-8 -*-
"""Interface for reinforcement learning. 
"""
import pdb

import random
import numpy as np
import pandas as pd
from backtester.stockdata import YahooData, Indicators, BaseData, TableData
from datasets.fakedata import FakeData

from backtester.backtest import Backtest, Strategy
from numpy.random import default_rng, Generator
from typing import NamedTuple
from backtester.definitions import DF_ADJ_CLOSE
from backtester.indicators import TrailingStats

class EnvStrategy(Strategy):
    """Simple strategy to interface with reinforcement Q-learning with 
    2 actions -- Buy/hold or Sell/Close"""
    
    def init(self):
        self.action = 0
        self.is_holding = False
        self.symbol = ''

    
    def next(self):
        if self.is_holding:
            if self.action == 0:
                self.sell_percent(self.symbol, 1.0)
                self.is_holding = False
                # print('selling')
        else:
            if self.action == 1:
                self.buy(self.symbol, self.available_funds)
                self.is_holding = True
                # print('buying')
    
    
    
class ObservationSpace(NamedTuple):
    shape : tuple
    
    
class ActionSpace:
    def __init__(self, n: int):
        self.n = n
        self._choices = np.arange(n)
        
        
    def set_rng(self, rng):
        self.rng = rng
        
        
    def sample(self):
        return self.rng.choice(self._choices)
    

    
    
class EnvBase:
    """Simple reinforcement training environment for a bot with 2 actions:
        - buy/hold using a stock.
        - sell/short a stock.
        
    This bot can only play on one stock at a time. 
    """
    def __init__(self, 
                 stock_data: BaseData,
                 indicators: Indicators,
                 start_date: np.datetime64 = None,
                 end_date: np.datetime64 = None,
                 price_name = DF_ADJ_CLOSE,
                 commission=.002,
                 game_length=30,
                 seed=0,):
        
        
        self.indicators = indicators
        self.commission = commission
        self.price_name = price_name        
        self.stock_data = stock_data
        
        stock_names = stock_data.symbol_names
        name1 = stock_names[0]
        
        df = indicators.dataframes[name1]
        col_num = df.values.shape[1]
        self.observation_space = ObservationSpace(shape=(col_num,))
        self.action_space = ActionSpace(n=2)        
        self._rng : Generator
        

        self._stock_names = stock_names
        self.game_length = game_length
        self.seed(seed)
        
        
    def step(self, action):
        strategy = self.backtest.strategy
        strategy.action = action
        self.backtest.step()
        self.time_index += 1
        return self._observe(), self._reward(), self._done(), self._info()
        
    
    def reset(self):
        rng: Generator = self._rng
        stock_data = self.stock_data
        
        self.symbol = rng.choice(self._stock_names)
        
        times =  stock_data.dataframes[self.symbol].index.values
        tnum = len(times) - self.game_length
        i1 = rng.integers(0, tnum)
        i2 = i1 + self.game_length
        self.time_index = i1
        self.stop_index = i2
        self._prices = stock_data.dataframes[self.symbol][self.price_name].values
        
        
        self.backtest = Backtest(stock_data=stock_data,
                                 strategy=EnvStrategy, 
                                 cash=100.0,
                                 commission=self.commission, 
                                 start_date=times[i1],
                                 end_date=times[i2],
                                 price_name=self.price_name
                                 )
        self.backtest.strategy.symbol = self.symbol
        self.backtest.step()
        self._assets = 100.0
        return self._observe()
    
        
    def _observe(self):
        """Observe latest indicators."""
        table: TableData = self.indicators.tables[self.symbol]
        return table.values[self.time_index]
      
    
    def _reward(self):
        """Calculate the reward of the latest step."""
        funds = self.backtest.strategy.available_funds
        try:
            assets = self.backtest.strategy.asset_values.values[0]
        except IndexError:
            assets = 0
            
        new_assets = funds + assets
        
        reward = new_assets - self._assets
        self._assets = new_assets
        return reward
    
    def _done(self):
        if self.time_index >= self.stop_index:
            return True
        else:
            return False
        
        
    def _info(self):
        date = self.backtest.strategy.date
        price = self._prices[self.time_index]
        new = {}
        new['date'] = date
        new['price'] = price
        new['symbol'] = self.symbol
        return new
        
        
    
    def seed(self, seed=None):
        self._rng = default_rng(seed)
        self._prng = random.Random(seed)
        self.action_space.set_rng(self._rng)
        

    
    
# class EnvSin30(EnvCycle30):
    





        
def env_sin20_growth():   
    """Test environment for simple, smooth sin wave with trailing growth rate
    as indicators."""
    def indicator1(df):
        close = df['Close']
        windows = [5, 10, 20, 50]
        outs = {}
        for window in windows:
            ts = TrailingStats(close, window)
            out = ts.exp_growth
            
            # Replace NAN with 0
            out[np.isnan(out)] = 0
            outs[f'{window}'] = out
        return outs
    
    fdata = FakeData(0, include=['Sin(20)'])
    indicators = Indicators(fdata)
    indicators.create(indicator1, name='growth')
    
    
    env = EnvBase(stock_data=fdata,
                  indicators=indicators, 
                  price_name='Close')
    env.reset()
    return env


def env_noise():
    def indicator1(df):
        close = df['Close']
        windows = [5, 10, 20, 50, 100]
        outs = {}
        for window in windows:
            ts = TrailingStats(close, window)
            out = ts.exp_growth
            out[np.isnan(out)] = 0
            outs[f'growth({window})'] = out
            
            out = ts.exp_reg_diff
            out[np.isnan(out)] = 0
            outs[f'diff({window})'] = out
        return outs
    fdata = FakeData(0, include=['Noise(1.0)'])
    indicators = Indicators(fdata)
    indicators.create(indicator1,)
    
    env = EnvBase(stock_data=fdata,
                  indicators=indicators, 
                  price_name='Close')
    env.reset()
    return env




# if __name__ == '__main__':
#     env = env_sin20_growth()
#     done = False
#     for ii in range(5000):
#         state, reward, done, _ = env.step(1)
#         print(ii, reward)
#         if done:
#             env.reset()
    
    
    
    
    
    