
# -*- coding: utf-8 -*-
import pdb

import pandas as pd
import numpy as np

from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats
from backtester.smooth import TrailingSavGol
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData
from backtester.exceptions import NotEnoughDataError
import datetime

def func1(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingSavGol(v, window_size)
    return t.velocity

def func2(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingSavGol(v, window_size)
    return t.acceleration


def under_value(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    return -t.exp_reg_diff

stocks = ['SPY', 
          # 'GOOG',
           'MSFT',
          # 'AAPL',
          # # 'TSLA',
           'AIG',
           'ALK',
           'GRA',
           'HAL',
           'CR',
          ]

yahoo_data = YahooData()
all_names = yahoo_data.get_symbol_names()
np.random.seed(0)
np.random.shuffle(all_names)
stocks1 = all_names[0 : 16]
stocks = []
for stock in stocks1:
    if len(yahoo_data[stock]) > 0:
        stocks.append(stock)
yahoo_data = YahooData(symbols=stocks)


class Strat1(Strategy):
    
    def init(self):
        # Build long and short rate metrics
        self.rate = self.indicator(func1, 301, name='rate')
        self.accel = self.indicator(func2, 301, name='accel')

        
        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stock = stocks[0]
        return
    
    
    def next(self):
        
        metrics = []
        delisted = self.sell_delisted()
        if len(delisted) > 0:
            self.current_stock = ''
        
        stocks = self.existing_symbols
        
        for stock in stocks:
            try:
                rate = self.rate(stock)
            except NotEnoughDataError:
                rate = np.array([-10])
            
            rate_last = rate[-1]
            rate_std = np.nanstd(rate)
            
            metric = rate_last 
            metrics.append(metric)
            
        metrics = np.array(metrics)
        imax = np.argmax(metrics)
        stock = stocks[imax]
        
        # Check if short term growth exists. 
        if True:
            if stock != self.current_stock:
                self.sell_percent(self.current_stock, amount=1.0)
                action = self.buy(stock, self.available_funds)
                self.current_stock = stock
                # print(action)
        else:
            # Sell if everything going down.
            self.sell_percent(self.current_stock, amount=1.0)
        return
    
    
if __name__ == '__main__':
    
    bt = Backtest(
        stock_data=yahoo_data, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2015, 4, 1),
        end_date=datetime.datetime(2020, 4, 26),
        )
    bt.start()
    perf = bt.stats.performance
    assets = bt.stats.asset_values
    exp  = bt.stats.exposure_percentages
    
    
    
    
    
    # -*- coding: utf-8 -*-

