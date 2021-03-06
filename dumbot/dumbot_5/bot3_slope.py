# -*- coding: utf-8 -*-

"""Bot formulated using 3-year Sharpe/Sortino Ratio. Works alright. 

"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb
from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats, append_nan
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData, TableData, Indicators

from backtester.exceptions import NotEnoughDataError
# from datasets.periodic_stats import read_rolling_stats
from backtester.utils import InterpConstAfter

from functools import cached_property
import datetime
import globalcache

import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()



# %% Get stock data

yahoo = YahooData()
symbols = yahoo.symbol_names
cache = globalcache.Cache(globals())

# rs = np.random.default_rng(0)
# rs.shuffle(symbols)




def post1(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    window = 252*5
    ts = TrailingStats(series, window)
    stat1 = ts.exp_growth
    if len(stat1) == 0:
        out = np.zeros(len(series))
        out[:] = np.nan
        return out
    return append_nan(stat1, window)




STOCKS = symbols
# STOCKS.append('SPY')
# STOCKS.append('VOO')
# STOCKS.append('GOOG')
# STOCKS.append('TSLA')
# STOCKS = np.array(STOCKS)[0:00]
rng = np.random.default_rng(1)
rng.shuffle(STOCKS)
STOCKS = STOCKS[0:200]

@cache.decorate
def create_indicator():
    yahoo.symbols = STOCKS
    indicator = Indicators(yahoo)
    indicator.create(post1)
    for key in indicator.dataframes.keys():
        indicator.dataframes[key]

indicator = create_indicator()
indicator.


# %% Strategies

class StopLoss:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self._equity_history = []
        
        
    
    
    def _get_equity(self):
        equity = self.strategy.state.equity
        self._equity_history.append(equity)
        return equity
    


class Strat1(Strategy):
        
    def init(self):
        # Build long and short rate metric
        self.growths = []

        # Arbitrarily set initial stock. Doesn't matter. 
        self.current_stocks = set()
        # self.stats = RollingStats()
        self.indicator1 = self.indicator(post1, )
        self.ii = 0
        self.loss_days = 0        
        return
    
    
    def next(self):
        self.ii += 1
        
        # if self.stop_loss():
        #     self.sell_all()
        #     return
        
        if self.ii % 5 != 0:
            stocks = self.get_stocks()
            self.buy_equally(stocks)
        
        
    def stop_loss(self):
        """Return True to stop tradeing. False to resume trading."""
        loss = table_stoploss.array_at(self.date)[0]
        if loss < 0:
            return True
        else: 
            return False
        
        
    def sell_all(self):
        if len(self.current_stocks) == 0:
            return
        
        print(self.date, 'Stop loss')
        for stock in self.current_stocks:
            self.sell_percent(stock, amount=1.0)
        self.current_stocks = set()
        
        
    def buy_equally(self, symbols: set[str]):
        current_stocks = self.current_stocks.copy()
        if len(symbols) == 0:
            return
        
        if len(symbols) == len(current_stocks):
            if symbols == current_stocks:
                return
        
        
        assets = self.state.asset_values
        commission = self._transactions.commission
        
        equity = self.state.equity
        
        
        stocks_to_sell = current_stocks - symbols
        stocks_to_rebalance = symbols & current_stocks
        stocks_to_buy =  symbols - current_stocks
        
        curr_to_change = stocks_to_rebalance | stocks_to_sell
        curr_to_change = list(curr_to_change)
        # pdb.set_trace()
        fee = assets[list(stocks_to_sell)].sum() * commission
        # fee2 = assets[list(stocks_to_buy)].sum() * commission
        # fee = fee1 + fee2 
        
        # Stock asset value to set portfolio at.
        set_value = (equity - fee) / len(symbols)
        

        # Sell
        for ii, stock in enumerate(stocks_to_sell):
            self.sell_percent(stock, amount=1.0)

            
            

        # # Rebalance
        rebalance_delta = set_value - assets[stocks_to_rebalance]
        
        # Rebalance - Sell
        for stock in stocks_to_rebalance:
            delta = rebalance_delta[stock]
            if delta < 0:
                self.sell(stock, amount=-delta)
                
        # Rebalance - Buy
        
        stocks_to_rebuy = rebalance_delta[rebalance_delta > 0].index
        num_buy = len(stocks_to_buy) + len(stocks_to_rebuy)
        
        # Recalculate available funds or equity
        cash = self.state.available_funds
        rebuy_assets = assets[stocks_to_rebuy].sum()
        rebuy_equity = cash + rebuy_assets

        equity_per_buy = rebuy_equity / num_buy
        rebalance_buy = equity_per_buy - assets[stocks_to_rebalance]

        for stock in stocks_to_rebuy:
            delta = rebalance_buy[stock]
            self.buy(stock, amount=delta)       
                
        # Buy
        for stock in stocks_to_buy:
            # try:
                self.buy(stock, amount=equity_per_buy)
            # except:
            #     pdb.set_trace()
        print(self.date)
        print(self.state.current_stocks)
        self.current_stocks = symbols
        # pdb.set_trace()
        # print(self.state.available_funds)
        if self.state.available_funds > 1:
            raise ValueError('Something went wrong with this calculation.')
        return
    
    
    def get_stocks(self):
        MAX_ALLOWED = 6
        metric_spy = self.indicator1.array('SPY')[-1, -1]
        metrics = [self.indicator1.array(stock)[-1, -1] for stock in STOCKS]
        metrics = np.array(metrics) - metric_spy
        
        
        # metrics = [self.stats.get(stock, self.date) for stock in STOCKS]
        isort = np.argsort(metrics)
        buy_indices1 = metrics > 0
        buy_indices2 = isort <= MAX_ALLOWED
        buy_indices = buy_indices1 & buy_indices2
        
        pdb.set_trace()
        new_stocks = STOCKS[buy_indices]
        return set(new_stocks)
        
    
if __name__ == '__main__':

    bt = Backtest(
        stock_data=yahoo, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2018, 6, 19),
        end_date=datetime.datetime(2021, 6, 25),
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