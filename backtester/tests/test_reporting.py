# -*- coding: utf-8 -*-

import pdb
import pandas as pd
import numpy as np

from backtester.reporting.snapshot import StockSnapshot
from backtester.backtest import Strategy

from backtester.backtest import Strategy, Backtest
from backtester.stockdata import YahooData, Indicators, DictData
from backtester.indicators import TrailingStats, append_nan
from backtester.definitions import DF_ADJ_CLOSE


from bokeh.plotting import figure, show

symbols = ('SPY', 'GOOG', 'MSFT', 'AAPL', 'TSLA')
indicator_keys = ['RoR-10', 'RoR-20', 'RoR-30']
yahoo = YahooData(symbols=symbols)
                  
                  
                  
def indicator10(df: pd.DataFrame, window_size=10):
    """Indicator construction function."""
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, window_size)
    y = ts.return_ratio
    x = ts.times
    new_series = pd.Series(y, index=x, name=f'RoR-{window_size}')
    new_df = pd.DataFrame(new_series)
    return new_df




class Strat1(Strategy):
    def init(self):
        self.indicators.from_func(indicator10, key='indicator-10')
        self.indicators.from_func(indicator10, 20, key='indicator-20')
        self.indicators.from_func(indicator10, 200, key='indicator-200')
    def next(self):
                
        if self.days_since_start <= 0:
            self.buy('SPY', 50)
            self.buy('GOOG', 25)
         
        
        snapshot = StockSnapshot(self)
        # snapshot.bar_pyplot()
        show(snapshot.bar_bokeh())
        pdb.set_trace()
    
    
    
    
    
date1 = np.datetime64('2016-01-01')
date2 = np.datetime64('2016-01-20')
bt = Backtest(
    stock_data = yahoo,
    cash = 100,
    strategy = Strat1,
    start_date = date1,
    end_date = date2,
    )

bt.run()