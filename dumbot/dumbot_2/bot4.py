# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb
from typing import Callable
from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData, TableData
from backtester.exceptions import NotEnoughDataError

from functools import cached_property
import datetime


import matplotlib.pyplot as plt


def my_indicator(df):
    series = df[DF_ADJ_CLOSE]
    # windows = [20, 50, 70, 90, 600]
    windows = [400, 600, 1200]
    windows = np.array(windows)
    weights = np.sqrt(windows) / windows[-1]
    weights = weights[:, None]
    

    
    # pdb.set_trace()
    growths = []
    for window in windows:
        ts = TrailingStats(series, window)
        g1 = ts.exp_growth
        g1[np.isnan(g1)] = -1
        growths.append(ts.exp_growth)
        
    growths = np.array(growths) #* weights
    out = growths.max(axis=0) 
    # out[max_loss > .15] = 0
    return out


yahoo = YahooData()

symbols = yahoo.get_symbol_names()
rs = np.random.default_rng(1)
rs.shuffle(symbols)

STOCKS = symbols[0:50]
STOCKS.append('SPY')
STOCKS.append('GOOG')
try:
    STOCKS.remove('TSLA')
except ValueError:
    pass
STOCKS = np.array(STOCKS)

class Strat1(Strategy):
        
    def init(self):
        # Build long and short rate metric
        self.growths = []

        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stocks = []
        self.indicator1 = self.indicator(my_indicator, name='metric')
        
        return
    
    
    def next(self):
        metrics = [self.indicator1.get_last(stock, -1) for stock in STOCKS]
        metrics = np.array(metrics)
        m_min = np.min(metrics[metrics > 0])
        m_range = np.max(metrics) - m_min
        
        
        
        p90th = metrics > (m_min + 0.7 * m_range)
        new_stocks = STOCKS[p90th]  
        for old_stock in self.current_stocks:  
            self.sell_percent(old_stock, amount=1.0)
        
        
        new_amount = self.available_funds / len(new_stocks)
        for new_stock in new_stocks:
            action = self.buy(new_stock, new_amount)
            # print(action)
        self.current_stocks = new_stocks
    
    
if __name__ == '__main__':

    bt = Backtest(
        stock_data=yahoo, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2016, 4, 1),
        end_date=datetime.datetime(2016, 4, 10),
        )
    bt.run()
    perf = bt.stats.performance
    my_perf = perf['equity'].values[-1]
    print('My performance', my_perf)
    assets = bt.stats.asset_values