
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats, trailing_percentiles
from backtester.smooth import TrailingSavGol
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData
import datetime

def func1(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    max_loss = t.max_loss
    tp = trailing_percentiles(max_loss, window=500)
    return tp


def func2(window_size, df: pd.DataFrame):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    max_loss = t.max_gain
    tp = trailing_percentiles(max_loss, window=500)
    return tp

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
        self.loss = self.indicator(func1, 30, name='loss-percentile')
        self.gain = self.indicator(func2, 30, name='gain-percentile')
        self.symbol = 'MSFT'
        
        # Arbitrarily set initial stock. Doesn't matter. 
        self.holding = False
        return
    
    
    def next(self):
        
        loss_percentile = self.loss('MSFT')[-1]
        gain_percentile = self.loss('MSFT')[-1]
        if loss_percentile > 95:
            if self.holding:
                self.holding = False
                action = self.sell_percent(self.symbol, 1)
                print(action)
        elif gain_percentile > 40:
            if not self.holding:
                self.holding = True
                action = self.buy(self.symbol, self.available_funds)
                print(action)

        return
    
    
if __name__ == '__main__':
    y = YahooData(symbols=stocks)
    
    bt = Backtest(
        stock_data=y, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2005, 4, 1),
        end_date=datetime.datetime(2009, 4, 26),
        )
    bt.start()
    perf = bt.stats.performance
    assets = bt.stats.asset_values
    exposures = bt.stats.exposure_percentages
        
    
    
    
    
    # -*- coding: utf-8 -*-

