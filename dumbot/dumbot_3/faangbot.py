# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import numpy as np
import pdb
from typing import Callable
from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats, cumulative_mean, cumulative_std
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData, TableData, Indicators
from backtester.exceptions import NotEnoughDataError



faang_stocks = ['AAPL', 'GOOG', 'GOOGL', 'NFLX', 'AMZN', 'FB']

symbols = faang_stocks + ['SPY']
slen = len(symbols)
yahoo = YahooData(symbols)


class Strat1(Strategy):
        
    def init(self):

        self.current_stocks = []
        self.ii = 0
        return
    
    
    def next(self):
                
        # MAX_ALLOWED = 30

        new_stocks = self.market_state.existing_symbols
        
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
        
        if len(to_buy) > 0:
            new_amount = self.state.available_funds / len(to_buy)
            for new_stock in to_buy:
                self.buy(new_stock, new_amount)
                
                
    
    
if __name__ == '__main__':

    bt = Backtest(
        stock_data=yahoo, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2012, 1, 1),
        end_date=datetime.datetime(2021, 2, 1),
        )
    bt.run()
    perf = bt.stats.performance
    my_perf = perf['equity'].values[-1]
    my_bench = bt.stats.benchmark('SPY')
    print('My performance', my_perf)
    print('SPY performance', my_bench)
    
    assets = bt.stats.asset_values