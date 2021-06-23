# -*- coding: utf-8 -*-

"""Bot formulated using 3-year Sharpe/Sortino Ratio. Works alright. 

"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb
from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData, TableData, Indicators
from backtester.exceptions import NotEnoughDataError
from datasets.periodic_stats import read_rolling_stats
from backtester.utils import InterpConstAfter

from functools import cached_property
import datetime


import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()




yahoo = YahooData()
symbols = yahoo.symbol_names
# rs = np.random.default_rng(0)
# rs.shuffle(symbols)


df_stats = read_rolling_stats()
df = df_stats['sortino']

class RollingStats:
    def __init__(self):
        stats_dict = read_rolling_stats()
        self.df = stats_dict['sharpe']
        self._interp_dict = {}   
    
    
    def get(self, symbol: str, date: np.datetime64):
        series = self.df[symbol]
        
        try:
            interp = self._interp_dict[symbol]
        except KeyError:
            interp = InterpConstAfter(series.index, series.values, before=-1)
            self._interp_dict[symbol] = interp
            
            
        out = interp.scalar(date)
        if np.isnan(out):
            out = -1
        return out


STOCKS = symbols
STOCKS.append('SPY')
STOCKS.append('VOO')
STOCKS.append('GOOG')
STOCKS.append('TSLA')
STOCKS = np.array(STOCKS)
rng = np.random.default_rng()
# rng.shuffle(STOCKS)
# STOCKS = STOCKS[0:500]



# %%

class Strat1(Strategy):
        
    def init(self):
        # Build long and short rate metric
        self.growths = []

        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stocks = []
        self.stats = RollingStats()
        self.ii = 0
        return
    
    
    def next(self):
        # growths = table.array_before(self.date)[-1]
        self.ii += 1
        
        if self.ii % 5 != 0:
            return
        
        MAX_ALLOWED = 6
        metrics = [self.stats.get(stock, self.date) for stock in STOCKS]
        metrics = np.array(metrics)
        isort = np.argsort(metrics)
        buy_indices1 = metrics > 0
        buy_indices2 = isort <= MAX_ALLOWED
        buy_indices = buy_indices1 & buy_indices2
        
        new_stocks = STOCKS[buy_indices]
        # if len(new_stocks) == 0:
        #     new_stocks = np.array(['SPY'])

        if len(new_stocks) == len(self.current_stocks):
            if np.all(new_stocks == self.current_stocks):
                return
            
        
        
        for old_stock in self.current_stocks:
            if old_stock in new_stocks:
                pass
            else:
                self.sell_percent(old_stock, amount=1.0)
                
                
        to_buy = []
        for new_stock in new_stocks:
            if new_stock not in self.current_stocks:
                to_buy.append(new_stock)
        if self.state.available_funds > 1e-4:
            if len(to_buy) == 0:
                to_buy = self.current_stocks
                        
        if len(to_buy) > 0:
            
            new_amount = self.state.available_funds / len(to_buy)
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
        start_date=datetime.datetime(2001, 1, 1),
        end_date=datetime.datetime(2021, 5, 20),
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
    
    s = yahoo.dataframes['SPY'][DF_ADJ_CLOSE]
    
    plt.semilogy(assets.index, assets['equity'], label='Algo')
    plt.semilogy(s, label='SPY')
    plt.legend()
    plt.grid()