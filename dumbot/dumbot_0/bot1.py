"""Simple bot that picks stocks with good average previous growth.
It works OK. Has potential."""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData
import datetime

def func1(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    return t.exp_growth


def under_value(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    return -t.exp_reg_diff

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


class Strat1(Strategy):
    
    def init(self):
        # Build long and short rate metrics
        self.short_rate = self.indicator(func1, 20, name='short_rate')
        self.long_rate = self.indicator(func1, 100, name='long_rate')
        self.under_value = self.indicator(func1, 50, name='under_value')
        
        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stock = stocks[0]
        return
    
    
    def next(self):
        
        metrics = []
        for stock in stocks:
            # print(stock)
            s = self.short_rate(stock)[-1]
            l = self.long_rate(stock)[-1]
            u = self.under_value(stock)[-1]
            metric = s + l + u
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
                print(action)
        else:
            # Sell if everything going down.
            self.sell_percent(self.current_stock, amount=1.0)
        return
    
    
if __name__ == '__main__':
    y = YahooData(symbols=stocks)
    
    bt = Backtest(
        stock_data=y, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2005, 4, 1),
        end_date=datetime.datetime(2020, 4, 26),
        )
    bt.start()
    perf = bt.stats.performance
    assets = bt.stats.asset_values
    
    
    
    
    
    