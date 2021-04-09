# -*- coding: utf-8 -*-

import datetime
import dataclasses
import operator


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy
import scipy.ndimage
from scipy.ndimage import uniform_filter

from sqlalchemy import create_engine

from dumbot.definitions import (
    CONNECTION_PATH,
    DF_DATE, DF_ADJ_CLOSE, DF_SMOOTH_CLOSE, DF_SMOOTH_CHANGE,
    DF_TRUE_CHANGE,
    ACTION_BUY, ACTION_SELL, ACTION_SELL_ALL, ACTION_HOLD
    )

from dumbot.stockdata.symbols import ALL
from dumbot.utils import get_rolling_average, read_dataframe, get_trading_days
from dumbot.utils import SymbolStats
from dumbot.dataclasses import Transactions, Action, SymbolTransactions

class __StockData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.df = (pd.read_sql(symbol, engine)
                     .set_index('Date', drop=True)
        )
        
        
    @property
    def close(self):
        return self.df['Adj Close']
    
    
    def get_range(self, days: int, end: datetime.date=None):
        """Get close dataframe for `days` interval, ending at `end`. 
        
        Paramters
        ---------
        days : int
            Number of days to fetch from end
            
        end : datetime
            Ending date. Default set to last time in data. 
            
        """
        if end is None:
            return self.close.iloc[-days : ]
        else:
            end = np.datetime64(end)
            df2 = self.close[self.df.index <= end]
            return df2.iloc[-days :]
        
        
    def smooth_range(self,
                     days: int, 
                     end: datetime.date=None,
                     window_size: int=21,
                     order: int=1):
        
        days2 = days + window_size
        series = self.get_range(days=days2, end=end)
        x = series.index
        y = series.values
        
        origin = int(window_size / 2)
        ys = y.copy()
        for m in range(1, order + 1):
            ys =  uniform_filter(ys,
                                 size=window_size,
                                 origin=origin,
                                 mode='constant')
        
        
        ydiff = np.diff(ys) / ys[1:]
        x = x[-days :] # - datetime.timedelta(days=origin)
        ys = ys[-days :]
        y = y[-days :]
        ydiff = ydiff[-days :]
        data = {'Close' : y, 
                'Close_Smooth' : ys,
                'Relative_Change' : ydiff }
        return pd.DataFrame(data=data, index=x)
        # return pd.Series(index=x, data=ys)
        

# def sign_change(a):
#     """https://stackoverflow.com/questions/2652368/
#     how-to-detect-a-sign-change-for-elements-in-a-numpy-array"""
    
#     return np.diff(np.sign(a))


    

        
    
class Bot1:
    def __init__(self, symbol: str, 
                 start: datetime.date,
                 stop: datetime.date,
                 history_start=None,
                 window_size=61):
        self.symbol = symbol
        
        
        if history_start is not None:
            history_start = np.datetime64(history_start)
        self.history_start = history_start
        self.start = np.datetime64(start)
        self.stop = np.datetime64(stop) 
        
        df = self._read_df()
        
        stat = SymbolStats(df=df, window_size=window_size)
        self.df = stat.true_rolling_avg()
        self.df = self._calculate_rolling_avg(df, window_size)
        self.window_size = window_size

        
        
        # day_buffer = datetime.timedelta(days=10)
        # self.trading_days = get_trading_days(start - day_buffer, None)
        self.active_days = get_trading_days(start, stop)
        
        
    def _read_df(self):
        df = read_dataframe(self.symbol)
        if self.history_start is not None:
            df = df[df.index >= self.history_start]
        
        return df
        
        
    def _calculate_rolling_avg(self, df, window_size):
        return get_rolling_average(df, window_size)
        
        
    def get_history(self, date: datetime.datetime):
        """Get history that the bot can see."""
        date = np.datetime64(date)
        return self.df[self.df.index <= date]
    
    
    def run(self):
        intial_funds = 100
        holding = False
        transactions = Transactions(intial_funds)
        self.transactions = transactions
        
        diff_threshold = 0.0
        for day in self.active_days:
            df = self.get_history(day)
            diff = df[DF_SMOOTH_CHANGE]
            
            noise = np.std(diff[-300:]) * 1
            # metric = np.mean(diff[-10:])
            # metric = -diff[-1]
            metric = diff[-1]
            
            
            if metric > -noise*.0 and not holding:
                funds = transactions.last_available_funds()
                action = Action(date=day, 
                                name=ACTION_BUY,
                                dollars = funds,
                                symbol=self.symbol,)
                transactions.add(action)
                funds = 0
                holding = True
                print(f'BUY {day} {action.gain:.2f}, '
                      f'funds={funds:.2f}, ' 
                      f'close={action.close:.2f}')        
                
            elif metric < -noise*3 and holding:
                action = Action(date=day, 
                                name=ACTION_SELL_ALL,
                                dollars = 0,
                                symbol=self.symbol)
                transactions.add(action)
                holding = False
                funds = transactions.last_available_funds()
                print(f'SELL {day} {action.gain:.2f}, '
                      f'funds={funds:.2f}, ' 
                      f'close={action.close:.2f}')
                
                a1 = transactions.balances[-2].net_assets
                a2 = transactions.balances[-1].net_assets
                ratio = a2/a1
                print(f'RATIO = {ratio:.2f}')
                
        # Add last hold action
        action = Action(date=self.active_days[-1], name=ACTION_HOLD, dollars=0)
        transactions.add(action)
        net_assets = transactions.last_net_assets()
        
        performance =  net_assets / intial_funds
        sp = self.get_benchmark()
        print(f'Bot performance = {performance:.2f}')
        print(f'Stock performance = {sp:.2f}')
        print(f'Net Assets = {net_assets:.2f}')
        
        
    def get_benchmark(self):
        st = self.transactions._symbol_transactions_dict[self.symbol]
        
        
        s1 = st.get_next_close(self.active_days[0])[0]
        s2 = st.get_next_close(self.active_days[-1])[0]         
        sp = s2 / s1
        print(f'start price = {s1}')
        print(f'end price = {s2}')
        
        return sp
    
    
    def hold_index(self):
        st = self.transactions._symbol_transactions_dict[self.symbol]
        st : SymbolTransactions
        return st.hold_index()
    
    
    def plot(self):
        dfs = self.get_history(self.df.index[-1])
        
        
        plt.subplot(2,1,1)
        plt.plot(self.df.index, self.df[DF_ADJ_CLOSE])
        plt.plot(dfs.index, dfs[DF_SMOOTH_CLOSE])
        
        hold_index = self.hold_index()

        
        df2 = self.df.loc[hold_index]
        plt.plot(df2.index, df2[DF_ADJ_CLOSE], '.g')
        # df3 = self.df.loc[sell_index]
        # plt.plot(df3.index, df3[DF_ADJ_CLOSE], '.r')    
        
        
        plt.subplot(2,1,2)
        plt.axhline(0, color='k')
        plt.plot(dfs.index, dfs[DF_SMOOTH_CHANGE], 'r')
        dfs3 = dfs.loc[hold_index]
        plt.plot(dfs3.index, dfs3[DF_SMOOTH_CHANGE], 'g')
        
        plt.plot(dfs.index, dfs[DF_TRUE_CHANGE], 'r--')
        plt.plot(dfs3.index, dfs3[DF_TRUE_CHANGE], 'g--')
        
    
    
# class __Bot1:
#     def __init__(self, symbol: str):
#         self.stock_data = StockData(symbol)
        
        
#     def run(self, days:int, end: datetime.date=None, window_size=21):
#         df_smooth = self.stock_data.smooth_range(
#             days=days,
#             end=end,
#             window_size=window_size
#         )
#         change = df_smooth['Relative_Change']
#         close = df_smooth['Close']
        
#         buy_locs = change > 0
#         sign_changes = np.diff(buy_locs.astype(int))
#         if buy_locs[0] == True:
#             sign_changes[0] = 1
            
#         action_locs = np.where(sign_changes != 0)[0]
#         signs1 = sign_changes[action_locs]
#         balance = 100
#         balance_history = [balance]
#         for loc, action in zip(action_locs, signs1):
#             date = df_smooth.index[loc]
#             # Signal a buy
#             if action == 1:
#                 start_price = close[loc]
#                 shares = balance / start_price
#                 balance_history.append(balance)
#                 print(f'{date}: buy at {start_price:.2f}, balance={balance:.2f}')
#             # Signal a sell
#             elif action == -1:
#                 end_price = close[loc]
#                 balance = shares * end_price
#                 balance_history.append(balance)
#                 print(f'{date}: sell at {end_price:.2f}, balance={balance:.2f}')
                
            
                
#         return signs1, balance_history
    
        
        
        
        
        
        
        
if __name__ == '__main__':
    symbols = np.array(ALL)
    # np.random.seed(0)
    np.random.shuffle(symbols)
    # symbol = 'VOO'
    
    symbol = symbols[34]
    print(symbol)
    start = datetime.date(2014, 4, 10)
    stop = datetime.date(2020, 6, 21)
    b = Bot1(symbol, start=start, stop=stop, window_size=21)
    b.run()
    b.plot()

 