"""
This bot seems to work!!!
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



yahoo = YahooData()
symbols = yahoo.get_symbol_names()
rs = np.random.default_rng(5)
rs.shuffle(symbols)

STOCKS = symbols[0:100]
STOCKS.append('SPY')
STOCKS.append('VOO')
# STOCKS.append('GOOG')
# STOCKS.append('TSLA')
STOCKS = np.array(STOCKS)


def post1(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, 252*5)
    stat1 = ts.exp_growth
    return stat1

yahoo.symbols = STOCKS
indicator = Indicators(yahoo)
indicator.create(post1)

df = indicator.get_column_from_all('post1()')
table = TableData(df)
index_spy = np.where(df.columns == 'SPY')[0][0]

class Strat1(Strategy):
        
    def init(self):
        # Build long and short rate metric
        self.growths = []

        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stocks = []
        
        return
    
    
    def next(self):
        growths = table.array_before(self.date)[-1]
        growths[np.isnan(growths)] = -1
        
        growth_spy = growths[index_spy]
        growth_delta = np.nanmax(growths) - growth_spy
        
        # MAX_ALLOWED = 30
        
        buy_indices = growths > growth_spy 
        # if np.sum(buy_indices) > MAX_ALLOWED:
        #     isort = np.argsort(growths)
        #     buy_indices = isort <  MAX_ALLOWED
            
        new_stocks = table.columns[buy_indices]
        
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
        start_date=datetime.datetime(2005, 1, 1),
        end_date=datetime.datetime(2021, 1, 1),
        )
    bt.start()
    perf = bt.stats.performance
    my_perf = perf['equity'].values[-1]
    my_bench = bt.stats.benchmark('SPY')
    my_bench2 = bt.stats.benchmark('VOO')
    print('My performance', my_perf)
    print('SPY performance', my_bench)
    print('VOO performance', my_bench2)
    
    assets = bt.stats.asset_values