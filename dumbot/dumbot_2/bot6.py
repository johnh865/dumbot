"""
This bot seems to work pretty well!!!

trailing stat return_ratio is sometimes better than exp_growth!
Actually Eh only sometimes...

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
rs = np.random.default_rng(1)
rs.shuffle(symbols)

STOCKS = symbols[0:600]
STOCKS.append('SPY')
STOCKS.append('VOO')
# STOCKS.append('GOOG')
# STOCKS.append('TSLA')
STOCKS = np.array(STOCKS)


# def post1(df:pd.DataFrame):
#     series = df[DF_ADJ_CLOSE]
#     ts = TrailingStats(series, 252*5)
#     stat1 = ts.exp_growth
#     return stat1

def post1(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, 252*5)
    stat1 = ts.return_ratio
    return stat1



yahoo.symbols = STOCKS
indicator = Indicators(yahoo)
indicator.create(post1)

df = indicator.get_column_from_all('post1()')
table = TableData(df)
index_spy = np.where(df.columns == 'SPY')[0][0]

# %%

class Strat1(Strategy):
        
    def init(self):
        # Build long and short rate metric
        self.growths = []

        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stocks = []
        self.ii = 0
        return
    
    
    def next(self):
        # growths = table.array_before(self.date)[-1]
        self.ii += 1
        
        if self.ii % 5 != 0:
            return
        
        
        
        growths = table.array_right_before(self.date)
        growths[np.isnan(growths)] = -1
        
        
        growth_spy = growths[index_spy]
        growth_delta = np.nanmax(growths) - growth_spy
        
        # MAX_ALLOWED = 5
        
        buy_indices = growths > max(0, growth_spy)
        
        mean2 = np.mean(growths[buy_indices])
        std2 = np.std(growths[buy_indices]) * 1.25
        buy_indices = growths > mean2 + std2
        
        
        # if np.sum(buy_indices) > MAX_ALLOWED:
        #     isort = np.argsort(growths)
        #     buy_indices = isort <  MAX_ALLOWED
            
        new_stocks = table.columns[buy_indices]
        # pdb.set_trace()
        if len(new_stocks) == len(self.current_stocks):
            if np.all(new_stocks == self.current_stocks):
                return
            

        # stock_values = self.asset_values
        # equity = np.sum(stock_values) + self.available_funds
        # new_amount = equity / len(new_stocks)
        
        
        for old_stock in self.current_stocks:
            if old_stock in new_stocks:
                pass
            else:
                self.sell_percent(old_stock, amount=1.0)
                
                
        to_buy = []
        for new_stock in new_stocks:
            if new_stock not in self.current_stocks:
                to_buy.append(new_stock)
        if self.available_funds > 1e-4:
            if len(to_buy) == 0:
                to_buy = self.current_stocks
                
                
        
        if len(to_buy) > 0:
            
            new_amount = self.available_funds / len(to_buy)
            for new_stock in to_buy:
                self.buy(new_stock, new_amount)


        self.current_stocks = new_stocks
        print(self.date, new_stocks)
    
    
if __name__ == '__main__':

    bt = Backtest(
        stock_data=yahoo, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2010, 1, 1),
        end_date=datetime.datetime(2020, 12, 20),
        )
    bt.run()
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
    print('My mean return', np.nanmean(r1))
    r2 = bt.stats.benchmark_returns('SPY', 252)
    print('SPY mean return', np.nanmean(r2))
    # plt.plot(perf.index, r1, label='Algo')
    # plt.plot(perf.index, r2, label='SPY')
    # plt.legend()
    # plt.grid()
    