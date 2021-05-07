"""

"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb
from typing import Callable
from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData, TableData, Indicators
from backtester.exceptions import NotEnoughDataError

from functools import cached_property
import datetime


import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()




yahoo = YahooData()
symbols = yahoo.get_symbol_names()
rs = np.random.default_rng(0)
rs.shuffle(symbols)

STOCKS = symbols[0:100]
STOCKS.append('SPY')
STOCKS.append('VOO')
# STOCKS.append('GOOG')
# STOCKS.append('TSLA')
STOCKS = np.array(STOCKS)



def post1(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, 15)
    stat1 = ts.return_ratio
    
    mean = np.nanmean(stat1)
    std = np.nanstd(stat1)
    
    stat2 = (stat1 - mean) / std
    return stat2 



yahoo.symbols = STOCKS
indicator = Indicators(yahoo)
indicator.create(post1)

df = indicator.get_column_from_all('post1()')


# %%
ii = df.values < -2.0
stocknum = ii.shape[-1]
isum = np.sum(ii, axis=1)
metric = isum / stocknum
df2 = pd.DataFrame(data=metric, columns=['metric'], index=df.index)




table = TableData(df2)


# %%

class Strat1(Strategy):
        
    def init(self):
        # Build long and short rate metric
        self.growths = []

        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stocks = []
        self.ii = 0
        self.hold = False
        return
    
    
    def next(self):
        metric = table.array_right_before(self.date)
        
        if self.hold:
            if metric >= .6:
                action = self.sell_percent('SPY', 1.0)
                self.hold = False
                print(action)
        else:
            if metric < .6:
                action = self.buy('SPY', self.available_funds)
                self.hold = True
                print(action)
            
        pass
    
 
    
    
if __name__ == '__main__':

    bt = Backtest(
        stock_data=yahoo, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2001, 1, 1),
        end_date=datetime.datetime(2021, 12, 20),
        )
    bt.start()
    perf = bt.stats.performance
    assets = bt.stats.asset_values
    my_perf = perf['equity'].values[-1]
    my_bench = bt.stats.benchmark('SPY')
    my_bench2 = bt.stats.benchmark('VOO')
    print('My performance', my_perf)
    print('SPY performance', my_bench)
    print('VOO performance', my_bench2)
    
    assets = bt.stats.asset_values
    
    
    # r1 = bt.stats.mean_returns(252)
    # r2 = bt.stats.benchmark_returns('SPY', 252)
    
    # plt.plot(perf.index, r1, label='Algo')
    # plt.plot(perf.index, r2, label='SPY')
    # plt.legend()
    # plt.grid()
    