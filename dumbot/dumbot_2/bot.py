# -*- coding: utf-8 -*-

"""Simple bot that picks stocks with good average previous growth.
It works OK. Has potential."""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb

from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData
from backtester.exceptions import NotEnoughDataError

import datetime


import matplotlib.pyplot as plt

def func1(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    return t.exp_growth

def func2(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    return t.exp_std_dev

stocks = ['SPY', 
          # 'GOOG',
           'MSFT',
          # 'AAPL',
          # # 'TSLA',
          # 'AIG',
          # 'ALK',
          # 'GRA',
          # 'HAL',
          # 'CR',
          ]


yahoo = YahooData()

stocks = yahoo.get_symbol_names()
rs = np.random.RandomState(10)
rs.shuffle(stocks)
stocks = stocks[0:20]
stocks = stocks + ['SPY']

class Strat1(Strategy):
    
    def init(self):
        # Build long and short rate metrics
        self.rate = self.indicator(func1, 321, name='rate')
        self.volatility = self.indicator(func2, 25, name='volatility')
        
        
        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stock = stocks[0]
        return
    
    
    def next(self):
        
        stocks1 = []
        for stock in stocks:
            try:
                rate = self.rate(stock)
                if len(rate) > 0:
                    stocks1.append(stock)
            except NotEnoughDataError:
                pass
                
              
        rates = [self.rate(stock) for stock in stocks1]        
        rates = np.concatenate(rates)
        r_mean = np.nanmean(rates)
        r_std = np.nanstd(rates)
        r_last = [self.rate(stock)[-1] for stock in stocks1]
        r_last = np.array(r_last)
        # Normalized rates w.r.t all stocks
        rates1 = (r_last - r_mean) / r_std
        
        
        volatilities1 = []
        for stock in stocks1:
            volatility = self.volatility(stock)
            v_mean = np.nanmean(volatility)
            v_std = np.nanstd(volatility)
            v_last = volatility[-1]
            volatility1 = (v_last - v_mean) / v_std
            volatilities1.append(volatility1)
        volatilities1 = np.array(volatilities1)
                  
        
        
        metrics = rates1 - np.maximum(volatilities1 - 0.5, 0) 
        metrics[np.isnan(metrics)] = -10
        # print(np.nanmax(rates1))
        imax = np.argmax(metrics)
        new_stock = stocks[imax]
        if new_stock != self.current_stock:

            self.sell_percent(self.current_stock, amount=1.0)
            action = self.buy(new_stock, self.available_funds)
            self.current_stock = new_stock
            print(action)

    
    
if __name__ == '__main__':
    y = YahooData(symbols=stocks)
    
    bt = Backtest(
        stock_data=y, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2008, 4, 1),
        end_date=datetime.datetime(2020, 4, 26),
        )
    bt.start()
    perf = bt.stats.performance
    assets = bt.stats.asset_values
    
    
    plt.semilogy(perf, alpha=.5)
    plt.semilogy(perf['equity'], '*')
    plt.legend(perf.columns)
    
    
    