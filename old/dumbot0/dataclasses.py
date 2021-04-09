# -*- coding: utf-8 -*-

import datetime
import dataclasses
import operator
from collections import deque 
from abc import abstractmethod, ABCMeta

from warnings import warn
from functools import cached_property

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy
import scipy.ndimage
from scipy.ndimage import uniform_filter

from sqlalchemy import create_engine

from dumbot.definitions import (CONNECTION_PATH, DF_ADJ_CLOSE, DF_DATE,
                                DF_HIGH, DF_LOW, DF_CLOSE)
from dumbot.definitions import (ACTION_BUY, ACTION_SELL, ACTION_DEPOSIT, 
                                ACTION_WITHDRAW, ACTION_HOLD,
                                ACTION_SELL_PERCENT)

from dumbot.stockdata.symbols import ALL
from dumbot.utils import read_dataframe



engine = create_engine(CONNECTION_PATH, echo=False)

@dataclasses.dataclass
class Action:
    """
    Parameters
    ----------
    date : datetime.datetime
        Date transaction executed.
    symbol : str or None
        stock market symbol, None for deposit/withdraw actions. 
    name : str
        Transaction type.
    amount : float
        Total dollar amount of transaction including fees. 
    fee : float
        Fees, commissions, etc of transaction.
        
    shares : float
        Number of shares after end of transaction
    gain : float
        Increase or decrease of amount at end of transaction
    price : float
        Price of stock symbol.
    
    """
    date : datetime.datetime
    name : str
    amount : float
    symbol : str = None


    shares : float = None
    fee : float = 0.0
    gain : float = 0.0
    price : float = None
    
    # available_funds : float = None
    # net_assets : float = None
    
    
    def __str__(self):
        date = self.date
        name = self.name
        symbol = self.symbol
        dstr = str(date).split('T')[0]
        
        base = f'Action(date={dstr}, name={name}, symbol={symbol}'
        if self.amount is not None:
            if self.amount > 0:
                base += f', amount={self.amount:.2f}'
        if self.price is not None:
            base += f', price={self.price:.2f}'
        if self.gain > 0:
            base += f', gain={self.gain:.2f}'
        base += ')'
        return base
    

@dataclasses.dataclass
class AccountBalance:
    """
    Parameters
    ----------
    date : datetime.datetime
        Date balance is calculated.
    available_funds : float
        Cash on hand
    equity : float
        Cash + assets value
    """
    date : datetime.datetime
    available_funds : float = None
    equity : float = None
    


class TradingError(Exception):
    pass



class SymbolTransactions:
    def __init__(self, symbol: str, actions: list[Action]=[], commission=0.0):
        self.symbol = symbol
        self.commission = commission
        self.queue = deque(actions)
        self.executed_actions = []
        self.df = read_dataframe(symbol)
        self.execute()
        
        
    # def _get_date_loc(self, date):
    #     date = np.datetime64(date)
    #     try:
    #         iloc = np.where(self.df.index == date)[0]
    #     except IndexError:
    #         raise ValueError(
    #             f'Date {date} was not found, probably not a trading day.'
    #             )
    #     return iloc
    
    def get_previous_trading_date(self, date):
        
        dates = self.df.index 
        ii = np.searchsorted(dates, date)
        return dates[ii - 1]
    
    
    def get_next_trading_date(self, date):
        """Get next available trading date. Return `date` if the input date
        is a trading date."""
        date = np.datetime64(date)
        dates = self.df.index 
        if date in dates:
            return date
        ii = np.searchsorted(dates, date)
        date1 = dates[ii]
        
        warn(
            f'Date {date} was not found, probably not a trading day.'
            f' New trade date set to {date1}.'
            )        
        return date1
    
    
    # def _get_previous_close(self, date):
    #     date = np.datetime64(date)
        
    #     try:
    #         out = self.df.loc[date]
    #     except KeyError:
    #         warn(
    #             f'Date {date} was not found, probably not a trading day.'
    #             )
    #         date = self.get_previous_trading_date(date)
    #         return self._get_previous_close(date)
        
    #     return out[DF_ADJ_CLOSE], date
    
    def _get_close(self, date):
        date = np.datetime64(date)
        return self.df.loc[date]
    
    
    def get_next_close(self, date):
        date = np.datetime64(date)
        try:
            out = self.df.loc[date]
        except KeyError:
            date1 = self.get_next_trading_date(date)
            
            warn(
                f'Date {date} was not found, probably not a trading day.'
                f' New date is {date1}.'
                )
            return self.get_next_close(date1)
        
        return out[DF_ADJ_CLOSE], date    
    
    
    
    def get_shares(self, date: datetime.datetime):
        """Get the number of invested shares for a given date."""
        # dates = [action.date for action in self.actions]
        actions = self.executed_actions
        
        if date < actions[0].date:
            return 0.0
        if len(actions) == 1:
            return actions[0].shares
        
        for ii in range(len(actions)-1):
            action1 = actions[ii]
            action2 = actions[ii + 1]
            if action1.date <= date < action2.date:
                return action1.shares
            
        
        # If date has gone through whole history, return last share value.
        return action2.shares
    
    
    def get_share_valuation(self, date: datetime.datetime):
        """Calculate value of shares at the given date."""
        close, date = self.get_next_close(date)
        shares = self.get_shares(date)
        value = close * shares
        return value
    
    
    def get_adj_price(self, date: datetime.datetime):
        """Estimate the price of the stock if bought in the date."""
        date = np.datetime64(date)
        out = self.df.loc[date]
        # try:
        #     out = self.df.loc[date]
        # except KeyError:
        #     raise TradingError(
        #         f'Date {date} was not found, probably not a trading day.'
        #         )        
            
        # close = out[DF_CLOSE]
        adj_close = out[DF_ADJ_CLOSE]
        # ratio = adj_close / close
        # high = out[DF_HIGH]
        # low = out[DF_LOW]
        # return (high + low) / 2.0 * ratio
        return adj_close 
    
    
    
    def _execute_one(self):
        action = self.queue.popleft()
        
        try:
            last_action = self.executed_actions[-1]
        except IndexError:
            shares = 0
        else:
            shares = last_action.shares
            
        date = action.date
        trade_date = self.get_next_trading_date(date)
        price = self.get_adj_price(trade_date)

        # close, date = self._get_last_close(date)
        amount = action.amount
        name = action.name    
    
        # Update action information
        action.price = price
        action.date = trade_date    
        
     
        if name == ACTION_BUY:
            fee = amount * self.commission
            shares += (amount - fee) / price
            action.shares = shares
            action.gain = -amount
            action.fee = fee
            
        elif name == ACTION_SELL:
            fee = amount * self.commission
            shares_sold = (amount - fee) / price
            shares += -shares_sold
            action.shares = shares
            action.gain = amount - fee
            action.fee = fee
            
            if shares < 0:
                raise TradingError(
                    f'{shares_sold: .2f} amount of shares sold exceeds'
                    f'{action.shares: .2f} amount owned.')
            
        elif name == ACTION_SELL_PERCENT:
            fraction = amount
            shares_sold = shares * fraction
            shares_kept = shares * (1 - fraction)
            sale_amount = shares_sold * price
            fee = sale_amount * self.commission
            
            action.shares = shares_kept
            action.gain = sale_amount - fee
            action.fee = fee
            
        self.executed_actions.append(action)
        return
    
    
    def execute(self):
        while self.queue:
            self._execute_one()            

    
    def add(self, action):
        self.queue.append(action)
        return self.execute()
    

    @cached_property
    def gain(self):
        return sum(action.gain for action in self.executed_actions)
        
        
    def hold_index(self):
        
        dates = self.df.index
        bools_list = []
        for action in self.executed_actions:
            if action.name == ACTION_BUY:
                date_start = action.date
                
            elif action.name == ACTION_SELL or action.name == ACTION_SELL_PERCENT:
                date_end = action.date
                bools = (dates >= date_start) & (dates <= date_end)
                bools_list.append(bools)
                
        # Handle with action ending with buy
        if action.name == ACTION_BUY:
            date_end = self.df.index[-1]
            bools = (dates >= date_start) & (dates <= date_end)
            bools_list.append(bools)

        bools2 = np.max(bools_list, axis=0)
        
        
        ihold = self.df.iloc[bools2].index
        # isell = self.df.iloc[~bools2].index
        return ihold
    
    
class Transactions:
    def __init__(self, init_funds=0, actions: list[Action]=None, 
                 commission=0.0):
        self.init_funds = init_funds
        self.commission = commission
        self._executed_actions = []
        self.balances = []
        if actions is None:
            self.queue = deque()
        else:
            self.queue = deque(actions)
            self.execute()

        
        return
    
    # def _execute_symbol_transactions(self):
    #     """Process symbol transactions."""
    #     sdict = {}
        
    #     for action in self.actions:
    #         name = action.name
    #         symbol = action.symbol
    #         if (name == ACTION_BUY or 
    #             name == ACTION_SELL or
    #             name == ACTION_SELL_ALL):
    #             if symbol in sdict:
    #                 sdict[symbol].append(action)
    #             else:
    #                 sdict[symbol] = [action]

    #     sdict2 = {}
    #     for symbol, actions in sdict.items():
    #         st = SymbolTransactions(symbol, actions)
    #         st.execute()
    #         sdict2[symbol] = st
            
    #     # Save the symbol transaction objects
    #     self._symbol_transactions_dict = sdict2
    #     return sdict2
    
    def _get_symbol_transactions_dict(self) -> dict[str, SymbolTransactions]:
        try:
            sdict = getattr(self, '_symbol_transactions_dict')
            return sdict

        except AttributeError:
            self._symbol_transactions_dict = {}
            return self._symbol_transactions_dict
    
    
    def _execute_symbol(self, action: Action):
        
        # Retrieve Symbol Transactions storage dict
        sdict = self._get_symbol_transactions_dict()
                
        name = action.name
        symbol = action.symbol
        
        if (name == ACTION_BUY or 
            name == ACTION_SELL or
            name == ACTION_SELL_PERCENT):
            if symbol in sdict:
                st = sdict[symbol]
            else:
                st = SymbolTransactions(symbol, commission=self.commission)
                sdict[symbol] = st
                
            # Execute using add.
            st.add(action)
        return  
                
                    
    def _execute_one(self):
        action = self.queue.popleft()

        try:
            last_balance: AccountBalance = self.balances[-1]
        except IndexError:
            available_funds = self.init_funds
        else:
            available_funds = last_balance.available_funds
        
        name = action.name
        date = action.date
        self._execute_symbol(action)


        if name == ACTION_DEPOSIT:
            available_funds += action.amount
            
        elif name == ACTION_WITHDRAW:
            available_funds += -action.amount
            
        elif name == ACTION_BUY:
            available_funds += -action.amount
            
        elif name == ACTION_SELL:
            available_funds += action.gain
            
        elif name == ACTION_SELL_PERCENT:
            available_funds += action.gain              
        
        elif name == ACTION_HOLD:
            pass
        
            
        balance = AccountBalance(
            date = date, 
            available_funds = available_funds,
            equity = None)
        
        if available_funds < 0:
            raise TradingError(f'Available funds is less than 0 from {action}!')
            
        self.balances.append(balance)
        self._executed_actions.append(action)
        self._execute_equity()
        return
    
        
        
    def execute(self):
        while self.queue:
            self._execute_one()            

    
    def add(self, action):
        self.queue.append(action)
        return self.execute()
    
    def buy(self, date: datetime.datetime, symbol: str, amount: float):
        action = Action(date, name=ACTION_BUY, symbol=symbol, amount=amount)
        return self.add(action)
    
    def sell(self, date: datetime.datetime, symbol: str, amount: float):
        action = Action(date, name=ACTION_SELL, symbol=symbol, amount=amount)
        return self.add(action)
    
    
    def sell_percent(self, date: datetime.datetime, symbol: str, amount: float):
        action = Action(date,
                        name=ACTION_SELL_PERCENT, 
                        symbol=symbol, amount=amount)
        return self.add(action)
    
    
    def hold(self, date: datetime.datetime):
        action = Action(date, name=ACTION_HOLD, amount=0.0)
        return self.add(action)
    

            
    # def execute(self):
    #     actions: list[Action] = sorted(self.actions, key=operator.attrgetter('date'))
    #     available_funds = self.init_funds
        
    #     # Process symbol transactions
    #     self._execute_symbol_transactions()
        
    #     balance_history = []
        
    #     # Process buy and sells on account balance
    #     for action in actions:
    #         name = action.name
    #         date = action.date            
            
    #         if name == ACTION_DEPOSIT:
    #             available_funds += action.dollars
                
    #         elif name == ACTION_WITHDRAW:
    #             available_funds += -action.dollars
                
    #         elif name == ACTION_BUY:
    #             available_funds += -action.dollars
                
    #         elif name == ACTION_SELL:
    #             available_funds += action.dollars
                
    #         elif name == ACTION_SELL_ALL:
    #             available_funds += action.gain
                
    #         balance = AccountBalance(
    #             date = date, 
    #             available_funds = available_funds,
    #             net_assets = None)
    #         balance_history.append(balance)
            
    #         action.available_funds = available_funds
    #         if available_funds < 0:
    #             raise TradingError(f'Available funds is less than 0 from {action}!')
        
    #     self.balances = balance_history
        
    #     # Calculate net assets for all transactions
    #     return
    
        
    def _execute_equity(self):
        """Calculate equity (cash + assets)"""
        sdict = self._get_symbol_transactions_dict()
        actions = self._executed_actions
        balances = self.balances
        
        for action, balance in zip(actions, balances):
            date = action.date
            equity = balance.available_funds
            
            for s_transaction in sdict.values():
                equity += s_transaction.get_share_valuation(date)
        
            balance.equity = equity
        return
    
    
    def last_equity(self):
        try:
            return self.balances[-1].net_assets
        except IndexError:
            return self.init_funds
        
        
    def last_available_funds(self):
        try:
            return self.balances[-1].available_funds
        except IndexError:
            return self.init_funds
    
    
                
class Strategy(metaclass=ABCMeta):
    def __init__(self, transactions: Transactions):
        self.transactions = transactions
        self.init()
        
        
    def _set_data(self, date, df: pd.DataFrame):
        self.data = df
        self.date = date
        
        
    @abstractmethod
    def init(self):
        pass
    
    @abstractmethod
    def next(self):
        pass
    
        
class StockData:
    def __init__(self, d=None):
        if d is not None:
            self.dict = d
        else:
            self.dict = {}
    def get_all_symbol_data(self, symbol:str):
        return self.dict[symbol]
    
    
    def get_symbol_data(self, symbol: str, date: datetime.datetime):
        date = np.datetime64(date)
        df = self.get_all_symbol_data(symbol)
        return df[df.index <= date]
           
    
    

class Backtest:
    def __init__(
            self,
            data: pd.DataFrame,
            strategy: Strategy,
            cash: float = 1.0,
            commission: float = .0,
            ):
        
        transactions = Transactions(init_funds=cash)
        
        
        
        
        