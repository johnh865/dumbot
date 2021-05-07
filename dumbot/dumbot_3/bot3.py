"""

"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb
from typing import Callable
from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats, cumulative_mean, cumulative_std
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData, TableData, Indicators
from backtester.exceptions import NotEnoughDataError

from functools import cached_property
import datetime
from numba import njit

import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()


@njit
def trough_volume(x: np.ndarray):
    current_max = x[0]
    current_max_loc = 0
    xlen = len(x)
    areas = np.zeros(xlen)
    
    # Record some information about each trough
    start_locs = np.zeros(xlen, dtype=np.int64)
    end_locs = np.zeros(xlen, dtype=np.int64)
    # min_locs = np.zeros(xlen, dtype=np.int64)
    areas_final = np.zeros(xlen)
    
    area = 0
    jj = 0
    for ii in range(xlen):
        xi = x[ii]
        
        # Price is rising
        if xi > current_max:
            start_locs[jj] = current_max_loc
            end_locs[jj] = ii
            areas_final[jj] = area
            jj += 1
            area = 0
            current_max_loc = ii
            current_max = xi
        # Trough detected
        else:
            area += (current_max - xi)
        
        areas[ii] = area    
        
    # If there's are left over record it for the final trough. 
    if area > 0:
        start_locs[jj] = current_max_loc
        end_locs[jj] = ii
        areas_final[jj] = area
        jj += 1
        
    start_locs = start_locs[0 : jj]
    end_locs = end_locs[0 : jj]
    areas_final = areas_final[0 : jj]
    return areas, start_locs, end_locs, areas_final


@njit
def peak_ratio(x: np.ndarray):
    current_max = x[0]
    current_max_loc = 0
    xlen = len(x)
    
    # Record some information about each trough
    start_locs = np.zeros(xlen, dtype=np.int64)
    end_locs = np.zeros(xlen, dtype=np.int64)
    pratios = np.zeros(xlen)
    
    area = 0
    jj = 0
    for ii in range(xlen):
        xi = x[ii]
        
        # Price is rising
        if xi > current_max:
            start_locs[jj] = current_max_loc
            end_locs[jj] = ii
            jj += 1
            area = 0
            current_max_loc = ii
            current_max = xi
            pratios[ii] = 0
        # Trough detected
        else:
            area += (current_max - xi)
            pratios[ii] = (current_max - xi) / current_max        
    return pratios

# %%


yahoo = YahooData()
symbols = yahoo.get_symbol_names()
rs = np.random.default_rng(0)
rs.shuffle(symbols)

# STOCKS = symbols[0:100]
# STOCKS.append('SPY')
# STOCKS.append('VOO')
# STOCKS.append('GOOG')
# STOCKS.append('TSLA')
STOCKS = ['SPY']
STOCKS = np.array(STOCKS)




def post1(df:pd.DataFrame):
    
    series = df[DF_ADJ_CLOSE]
    pratio = peak_ratio(series.values)
    pseries = pd.Series(data=pratio, index=series.index)
    
    ts = TrailingStats(pseries, 15)
    slope, interc, rvalue = ts.linear_regression
    r2 = rvalue**2
    
    slope_std = cumulative_std(slope)    
    r_std = cumulative_std(rvalue)
    
    d = {}
    d['metric'] = -slope * r2 / slope_std
    d['slope'] = slope
    d['r2'] = r2
    d['peak_ratio'] = pratio
    return d


def post2(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, 8)
    slope, interc, rvalue = ts.linear_regression
    r2 = rvalue**2
    metric = slope * r2
    
    std = cumulative_std(metric)
    return metric / std


yahoo.symbols = STOCKS
indicator = Indicators(yahoo)
indicator.create(post2, name='p')

# df = indicator.get_column_from_all('post1()')
df = indicator.get_symbol_all('SPY')
table = TableData(df)


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
            if metric <= -1:
                action = self.sell_percent('SPY', 1.0)
                self.hold = False
                print(action)
        else:
            
            if metric > -1:
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
    
    
    r1 = bt.stats.mean_returns(252)
    # r2 = bt.stats.benchmark_returns('SPY', 252)
    
    # plt.plot(perf.index, r1, label='Algo')
    # plt.plot(perf.index, r2, label='SPY')
    # plt.legend()
    # plt.grid()


# if __name__ == '__main__':
#     # plt.subplot(2,2,1)
#     # plt.plot(df["p['peak_ratio']"])
#     # plt.subplot(2,2,2)
#     # plt.plot(df["p['metric']"])
#     # plt.ylim(0, None)
    
#     test()
    
    
    
    
    
    