# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from stockdata import YahooData, BaseStockData, Indicators
from backtester.backtest import Backtest, Strategy
from numpy.random import default_rng, Generator
from typing import NamedTuple
from backtester.definitions import DF_ADJ_CLOSE


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
        else:
            if self.action == 1:
                self.buy(self.symbol, self.available_funds)
    
    
    
class ObservationSpace(NamedTuple):
    shape : tuple
class ActionSpace(NamedTuple):
    n : int
    
class EnvCycle30:
    
    
    
    def __init__(self, 
                 stock_data: BaseStockData,
                 start_date: np.datetime64 = None,
                 end_date: np.datetime64 = None,
                 price_name = DF_ADJ_CLOSE,
                 commission=.002,
                 game_length=30):
        
        self.backtest = Backtest(stock_data=stock_data,
                                 strategy=EnvStrategy,
                                 cash=100.0,
                                 commission=commission,
                                 start_date=start_date,
                                 end_date=end_date,
                                 )
        
        
        stocknames = stock_data.get_symbol_names()
        name1 = stocknames[0]
        df = stock_data.get_symbol_all(name1)
        col_num = df.values.shape[1]
        
        # Get column index for stock price. 
        col_price = list(df.columns).index(price_name)
        
        
        self.observation_space = ObservationSpace(shape=(col_num,))
        
        self.action_space = ActionSpace(n=2)
        self._rng : Generator
        self._length = len(self.backtest.active_days)
        self._num_stocks = len(stocknames)
        self._indicator_data = df.drop(price_name)
        
        
        self.game_length = game_length
        
        
    def step(self, action):
        strategy = self.backtest.strategy
        strategy.action = action
        
        
        self.backtest.step()
        
    
    def reset(self):
        rng: Generator = self._rng
        time_index = rng.integer(0, self._length - self.game_length)
        self.backtest.reset(time_index=time_index)
        
        
        
    def _observe(self):
        strategy = self.backtest.strategy
        strategy.stock_data
        

        
    
    
    def seed(self, seed=None):
        self._rng = default_rng(seed)
        
    
    
    def action_space(self):
        pass
        
        