# -*- coding: utf-8 -*-
import datetime
from abc import abstractmethod, ABCMeta
from typing import Callable 
from functools import cached_property

import numpy as np
import pandas as pd


from backtest.model import Transactions, SymbolTransactions, TransactionsDF
from backtest.utils import _delete_attr, dates2days, get_trading_days
from backtest.utils import interp_const_after
from backtest.stockdata import StockData, Indicators
from backtest.exceptions import NoMoneyError, TradingError
from backtest.definitions import SMALL_DOLLARS

class Strategy(metaclass=ABCMeta):
    """Base class for creating trading strategies."""
    
    def __init__(self, transactions: Transactions):
        self._transactions = transactions
        self._indicators = Indicators()        
        self.init()
        
        
    def _set_data(self, 
                  date: datetime.datetime,
                  days_since_start : int):
        """Set data to use for the current day for used in self.next."""
        transactions = self._transactions
        self._date = date   
        self._days_since_start = days_since_start
        
        # Clean cached property.
        _delete_attr(self, 'asset_values')
        _delete_attr(self, 'stock_data')


        
    def _delete_cache(self):
        _delete_attr(self, 'available_funds')
        _delete_attr(self, 'asset_values')

        
    @abstractmethod
    def init(self):
        """Definite your indicators here."""
        pass
    
    @abstractmethod
    def next(self):
        """Define your strategy per iteration here."""
        pass
    
    
    def _increment_small_time(self):
        """For multiple transactions, increment a tiny amount of time."""
        self._date += np.timedelta64(1, 's')
        
    
    
    def buy(self, symbol: str, amount: float):
        if amount > self.available_funds + SMALL_DOLLARS:
            raise NoMoneyError('Not enough funds for buy.')
            
        self._delete_cache()
        self._increment_small_time()
        trade = self._transactions.buy(
            date=self._date, symbol=symbol, amount=amount)
        return trade
        
        
    def sell(self, symbol: str, amount: float):
        """Sell a dollar amount of an asset."""
        
        self._delete_cache()
        self._increment_small_time()
        trade =  self._transactions.sell(
            date=self._date, symbol=symbol, amount=amount)
        return trade
        
        
    def sell_percent(self, symbol: str, amount: float):
        """Sell a percentage of an asset."""
        self._delete_cache()
        self._increment_small_time()
        trade = self._transactions.sell_percent(
            date=self._date, symbol=symbol, amount=amount)
        return trade
        
        
    @cached_property
    def available_funds(self):
        """Available cash for trading."""
        return self._transactions.get_available_funds(self._date)
    
    
    @cached_property
    def asset_values(self) -> pd.Series:
        """Pandas Series of asset value of each traded symbol."""
        return self._transactions.get_asset_values(self._date)
    
    
    @cached_property
    def stock_data(self):
        """dict[DataFrame] : Dataframes for each symbol."""
        return self._indicators.stock_data.get_symbols_before(self._date)  
    
    
    def indicator(self,
                  func: Callable, 
                  *args,
                  name: str=None, 
                  **kwargs) -> "_IndicatorValue":
        """Create an indicator and return a `_IndicatorValue

        Parameters
        ----------
        func : Callable
            Function to calculate indicator. `func` must accept a dataframe
            using keyword argument `df`:
                
            >>> value = func(*args, df=df, **kwargs)
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
            each stock symbol.
            
            .. code-block :: python
                        
                def init(self):
                    self.my_indicator = self.indicator(func1)
                    
                def next(self):
                    symbol = 'MSFT'
                    value = self.my_indicator(symbol)

        """
        """Create an indicator and return a `_IndicatorValue`."""
        name = self._indicators.create(func, *args, name=name, **kwargs)
        return _IndicatorValue(
            name=name,
            indicators=self._indicators,
            strategy=self)
    
    
    @property
    def date(self):
        return self._date
    
    
    @property
    def days_since_start(self):
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
        
        
    def __call__(self, symbol):
        """Retrieve indicator value for given symbol."""
        date = self.strategy.date
        return self.indicators.get_before(symbol, self.name, date)
    

class Backtest:
    """Main testing component to test and run strategies."""
    def __init__(self,
            stock_data: StockData,
            strategy: Strategy,
            cash: float = 1.0,
            commission: float = .0,
            start_date: datetime.date = None,
            end_date: datetime.date = None
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
        self.transactions.hold(self.active_days[-1])
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
    
            
       

            
            








