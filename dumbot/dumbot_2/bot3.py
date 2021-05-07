# -*- coding: utf-8 -*-

"""Attempt to optimize window. I got to optimum of window=321. This window
doesn't seem to generalize. """
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

def fn_growth(df: pd.DataFrame, window_size):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    return t.exp_growth

def fn_std_change(df: pd.DataFrame, window_size):
    v = df[DF_ADJ_CLOSE]
    t = TrailingStats(v, window_size)
    return t.exp_std_dev


yahoo = YahooData()

stocks = yahoo.get_symbol_names()
rs = np.random.RandomState(10)
rs.shuffle(stocks)
stocks = stocks[0:20]
stocks = stocks + ['SPY']

class Indicators:
    def __init__(self, stocks:list[str], func: Callable, fargs=(), fkwargs=None):
        self.stocks = stocks
        self.func = func
        self.fargs = fargs
        if fkwargs is None:
            fkwargs = {}
            
        self.fkwargs = fkwargs
        
        
    @cached_property
    def indicator_table(self) -> TableData:
        yahoo = YahooData()
        s_list = []
        for stock in self.stocks:
            df = yahoo[stock]
            out = self.func(df, *self.fargs, **self.fkwargs)
            series = pd.Series(data=out, index=df.index, name=stock)
            s_list.append(series)
            
        df = pd.concat(s_list, axis=1)
        return TableData(df)
    
    
    def _convert_to_table(self, array:np.ndarray):
        df = pd.DataFrame(
            data=array, 
            columns=self.stocks, 
            index=self.indicator_table.df.index)
        return TableData(df)
    
    
    @cached_property
    def normalize_by_all(self) -> TableData:
        """Normalize data by all stock data."""
        values = self.indicator_table.df.values
        index = self.indicator_table.df.index.values
        ilen = len(index)
        
        new = []
        for ii in range(ilen):
            values_ii = values[0 : ii]
            mean = np.nanmean(values_ii)
            std = np.nanstd(values_ii)
            new_ii = (values[ii] - mean) / std
            new.append(new_ii)
        return self._convert_to_table(np.array(new))
    
    
    @cached_property
    def normalize_by_each(self) -> TableData:
        """Normalize each symbol's data by its previous data."""
        values = self.indicator_table.df.values
        index = self.indicator_table.df.index.values
        ilen = len(index)
        
        new = []
        for ii in range(ilen):
            values_ii = values[0 : ii]
            mean = np.nanmean(values_ii, axis=0)
            std = np.nanstd(values_ii)
            new_ii = (values[ii] - mean) / std
            new.append(new_ii)
        return self._convert_to_table(np.array(new))        
        

        
STOCKS = stocks

class Strat1(Strategy):
        
    def init(self):
        # Build long and short rate metric
        self.growths = []
        windows = np.arange(10, 100, 10)
        for window in windows:
            indic = Indicators(
                STOCKS, 
                fn_growth, 
                fargs=(window,)).normalize_by_all
            self.growths.append(indic)
            
        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stock = STOCKS[0]
        return
    
    
    def next(self):
        
        growths = []
        for indic in self.growths:
            growth = indic.array_before(self.date)[-1]
            growth[np.isnan(growth)] = -10
            growths.append(growth)
        
        growths = np.array(growths)
        growths = np.mean(growths, axis=0)
        pdb.set_trace()
        imax = np.argmax(growths)
        new_stock = STOCKS[imax]
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
        start_date=datetime.datetime(2012, 4, 1),
        end_date=datetime.datetime(2014, 4, 26),
        )
    bt.run()
    perf = bt.stats.performance
    my_perf = perf['equity'].values[-1]
    print('My performance', my_perf)
    assets = bt.stats.asset_values
    
    
    # plt.semilogy(perf, alpha=.5)
    # plt.semilogy(perf['equity'], '*')
    # plt.legend(perf.columns)
    
    
    