# -*- coding: utf-8 -*-
import datetime
from abc import abstractmethod, ABCMeta
from typing import Callable 
from functools import cached_property

import numpy as np
import pandas as pd


from backtester.model import Transactions, SymbolTransactions, TransactionsDF
from backtester.utils import delete_attr, dates2days, get_trading_days
from backtester.utils import interp_const_after
from backtester.stockdata import BaseData, Indicators
from backtester.exceptions import NoMoneyError, TradingError
from backtester.definitions import SMALL_DOLLARS

SMALL_TIME_SECONDS = 1

class Strategy(metaclass=ABCMeta):
    """Base class for creating trading strategies.
    Use by creating a new strategy class by inheriting this baseclass. 
    
    Then define methods `init` and `next`. 
    
    - `init` initializing indicators and other object attributes.
    - `next` increments the simulation by one trading day. 
    """
    
    def __init__(self, transactions: Transactions):
        self._transactions = transactions
        self._indicators = Indicators() 
        self.init()
        
        
    # def _set_indicators(self, indicators: Indicators=None):
    #     """Set indicators."""
    #     if indicators is not None:
    #         self._indicators = indicators
        
        
    @abstractmethod
    def init(self):
        """Definite your indicators here."""
        raise NotImplementedError('This method needs to be implemented by user')
    
    @abstractmethod
    def next(self):
        """Define your strategy per increment here."""
        raise NotImplementedError('This method needs to be implemented by user')    
    
        
    def _set_data(self, 
                  date: np.datetime64,
                  days_since_start : int):
        """Set data to use for the current day for used in self.next."""

        self._date = date   
        self._days_since_start = days_since_start
        
        # Clean cached property.
        delete_attr(self, 'asset_values')
        delete_attr(self, 'stock_data')

        
    def _delete_cache(self):
        delete_attr(self, 'available_funds')
        delete_attr(self, 'asset_values')

    
    def _increment_small_time(self):
        """For multiple transactions, increment a tiny amount of time."""
        self._date += np.timedelta64(SMALL_TIME_SECONDS, 's')
        
    
    def buy(self, symbol: str, amount: float):
        """Buy a dollar amount of an asset during current increment."""
        if amount > self.available_funds + SMALL_DOLLARS:
            raise NoMoneyError('Not enough funds for buy.')
            
        self._delete_cache()
        self._increment_small_time()
        trade = self._transactions.buy(
            date=self._date, symbol=symbol, amount=amount)
        return trade
        
        
    def sell(self, symbol: str, amount: float):
        """Sell a dollar amount of an asset during current increment."""
        
        self._delete_cache()
        self._increment_small_time()
        trade =  self._transactions.sell(
            date=self._date, symbol=symbol, amount=amount)
        return trade
        
        
    def sell_percent(self, symbol: str, amount: float):
        """Sell a percentage of an asset during current increment."""
        self._delete_cache()
        self._increment_small_time()
        trade = self._transactions.sell_percent(
            date=self._date, symbol=symbol, amount=amount)
        return trade
        
        
    @cached_property
    def available_funds(self):
        """Available cash for trading during current increment."""
        return self._transactions.get_available_funds(self._date)
    
    
    @cached_property
    def asset_values(self) -> pd.Series:
        """Pandas Series of asset value of each traded symbol for current increment."""
        return self._transactions.get_asset_values(self._date)
    
    
    @cached_property
    def stock_data(self):
        """dict[DataFrame] : Dataframes for each symbol for the current increment."""
        return self._indicators.stock_data.get_symbols_before(self._date)  
    
    
    def indicator(self,
                  func: Callable, 
                  *args,
                  name: str=None, 
                  **kwargs) -> "_IndicatorValue":
        """Create an indicator and return a `_IndicatorValue.

        Parameters
        ----------
        func : Callable
            Function to calculate indicator. `func` must accept a dataframe
            using keyword argument `df`:
                
            >>> value = func(*args, df=df, **kwargs)
            
            df : pd.DataFrame
                Pandas dataframe of data retrieved from `StockData` object
                found in `Backtest`.
            value : dict or np.ndarray
                Indicator values for each row of dataframe. 
            
        *args : 
            Positional arguments for `func`.
        name : str, optional
            Name of indicator for dataframes. The default is None.
        **kwargs : 
            Keyword arguments for `func`.

        Returns
        -------
        _IndicatorValue
            Object used to retrieve indicator values in `self.next` for
            each stock symbol. For example:
            
            .. code-block :: python
                        
                def init(self):
                    self.my_indicator = self.indicator(func1)
                    
                def next(self):
                    symbol = 'MSFT'
                    value = self.my_indicator(symbol)

        """
        name = self._indicators.create(func, *args, name=name, **kwargs)
        return _IndicatorValue(
            name=name,
            indicators=self._indicators,
            strategy=self)
    
    
    @property
    def date(self):
        """The date of the current simulation increment."""
        return self._date
    
    
    @property
    def days_since_start(self):
        """Days since the beginning of simulation for the current simulation increment."""
        return self._days_since_start

    
    
class _IndicatorValue:
    """Store reference to indicator, retrieve the indicator data for 
    a given symbol."""
    def __init__(self, name: str, 
                 indicators: Indicators,
                 strategy: Strategy):
        self.name = name
        self.indicators = indicators
        self.strategy = strategy
        
        
    def __call__(self, symbol: str):
        """Retrieve indicator value for given symbol."""
        date = self.strategy.date
        dict1 = self.indicators.get_dict_before(symbol, date)
        return dict1[self.name]
    

class Backtest:
    """Main testing component to test and run strategies.

    Parameters
    ----------
    stock_data : BaseData
        Stock data.
    strategy : Strategy class
        Strategy class. Do not instantiate. 
    cash : float, optional
        Starting balance. The default is 1.0.
    commission : float, optional
        Fee ratio of each transaction from [0 to 1]. The default is 0.
    start_date : np.datetime64, optional
        Simulation start date. The default is None.
    end_date : np.datetime64, optional
        Simulation end date. The default is None.
        
        
    Attributes
    ----------
    active_days : np.ndarray[np.datetime64]
        Days in which to actively trade
    transations : Transactions
        Transactions generation object
    stats : BacktestStats
        Performance, balance, and transaction summary and statistics. 

            
            """
    def __init__(self,
            stock_data: BaseData,
            strategy: Strategy,
            cash: float = 1.0,
            commission: float = .0,
            start_date: np.datetime64 = None,
            end_date: np.datetime64 = None,
            ):

        self.stock_data = stock_data
        self.transactions = Transactions(
            stock_data = stock_data,
            init_funds = cash,
            commission = commission
        )
        self.strategy = strategy(self.transactions)
        self.start_date = start_date
        self.end_date = end_date
        
        dates = stock_data.get_trade_dates()
        self.active_days = get_trading_days(
            dates, 
            self.start_date,
            self.end_date
        )
        
        self._time_int = dates2days(self.active_days)
        return
    
    
    def start(self):
        self.strategy : Strategy
        self.strategy._indicators.set_stock_data(self.stock_data)
        self.strategy.init()
        self.transactions.hold(self.active_days[0])
        
        for ii, active_day in enumerate(self.active_days):            
            self.strategy._set_data(
                date = active_day,
                days_since_start = self._time_int[ii]
                )
            self.strategy.next()
            
        final_day = self.active_days[-1]
        final_day += np.timedelta64(SMALL_TIME_SECONDS, 's')
        self.transactions.hold(final_day)
        
        self.stats = BacktestStats(self)
        

    
class BacktestStats:
    """Process dataframes and information for transactions."""
    
    COLUMN_DATE = 'date'
    COLUMN_EQUITY = 'equity'
    COLUMN_FUNDS = 'available_funds'
    
    def __init__(self, backtest: Backtest):
        self._backtest = backtest
        self._transactions = backtest.transactions
        self._active_days = self._backtest.active_days
        
        
    @property
    def transactions(self):
        """Summary of trades and transactions."""
        return self._transactions.dataframe
        
    
    @cached_property
    def asset_values(self):
        """Get asset values for all available dates."""
        st : SymbolTransactions
        transactions = self._transactions
        dates = self._active_days
        
        # Get symbol valuations
        symbols = transactions.get_symbols_list()
        d = {}
        d[self.COLUMN_DATE] = dates
        for symbol in symbols:
            st = transactions.get_symbol_transactions(symbol)
            d[symbol] = st.get_share_valuation(dates)
            
        df = pd.DataFrame(d)
        df = df.set_index(self.COLUMN_DATE, drop=True)

        # Get cash and total equity
        df0 : TransactionsDF
        df0 = transactions.dataframe
        cash_a  = df0[self.COLUMN_FUNDS]
        dates_a = df0.index
                
        df[self.COLUMN_FUNDS] = interp_const_after(dates_a, cash_a, dates)
        df[self.COLUMN_EQUITY] = df.values.sum(axis=1)
        return df
 
    
    @cached_property
    def performance(self):
        """Normalized performance of bot vs considered stocks."""
        st : SymbolTransactions
        transactions = self._transactions
        symbols = transactions.get_symbols_list()
        dates = self._active_days
        
        d = {}
        d[self.COLUMN_DATE] = dates
        
        # Get stock price
        for symbol in symbols:
            st = transactions.get_symbol_transactions(symbol)
            price = st.get_adj_price(dates)
            d[symbol] = price / price[0]
        
        # Get our equity
        equity = self.asset_values[self.COLUMN_EQUITY]
        equity = equity / equity[0]
        d[self.COLUMN_EQUITY] = equity
        return pd.DataFrame(d)
    
            
       

            
            








