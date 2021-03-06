# -*- coding: utf-8 -*-
import pdb
from typing import Union, Hashable
import datetime
from abc import abstractmethod, ABCMeta
from typing import Callable 
from functools import cached_property, lru_cache
import warnings 

import numpy as np
import pandas as pd


from backtester.model import Transactions, SymbolTransactions, TransactionsDF
from backtester.model import Action
from backtester.model import TransactionsLastState, MarketState

from backtester.utils import delete_attr, dates2days, get_trading_days
from backtester.utils import interp_const_after
from backtester.stockdata import BaseData, Indicators, TableData, DictData
from backtester.exceptions import (
    NoMoneyError, TradingError, NotEnoughDataError, BacktestError)

from backtester.definitions import SMALL_DOLLARS, DF_ADJ_CLOSE
from backtester.indicators import TrailingStats, TrailingIntervals

SMALL_TIME_SECONDS = 1


class Strategy(metaclass=ABCMeta):
    """Base class for creating trading strategies.
    Use by creating a new strategy class by inheriting this baseclass. 
    
    Then define methods `init` and `next`. 
    
    - `init` initializing indicators and other object attributes.
    - `next` increments the simulation by one trading day. 
    """
    
    state : TransactionsLastState
    market_state : MarketState
    
    def __init__(self, transactions: Transactions):
        self._transactions = transactions
        self.indicators = IndicatorManager(self)
        return
    
        
        
    # def _set_indicators(self, indicators: Indicators=None):
    #     """Set indicators."""
    #     if indicators is not None:
    #         self._indicators = indicators
        
        
    @abstractmethod
    def init(self):
        """Define initialization here."""
        raise NotImplementedError('This method needs to be implemented by user')
    
    
    @abstractmethod
    def next(self):
        """Define your strategy per increment here."""
        raise NotImplementedError('This method needs to be implemented by user')    
        
        
    @property
    def date(self):
        return self._date
    
        
    def _set_data(self, 
                  date: np.datetime64,
                  days_since_start : int):
        """Set data to use for the current day for used in self.next."""

        self._date = date   
        self._days_since_start = days_since_start
        self.state = TransactionsLastState(self._transactions, date)
        self.market_state = MarketState(self._transactions, date)
        self.current_actions = []

        
    def _update_state(self):
        self.state = TransactionsLastState(self._transactions, self.date)
        
        return
    
    
    def _increment_small_time(self):
        """For multiple transactions, increment a tiny amount of time."""
        self._date += np.timedelta64(SMALL_TIME_SECONDS, 's')
        
    
    def buy(self, symbol: str, amount: float) -> Action:
        """Buy a dollar amount of an asset during current increment."""
        
        if amount > self.state.available_funds + SMALL_DOLLARS:
            raise NoMoneyError('Not enough funds for buy.')
            
        self._increment_small_time()
        trade = self._transactions.buy(
            date=self._date, symbol=symbol, amount=amount)
        self._update_state()
        self.current_actions.append(trade)
        
        return trade
        
        
    def sell(self, symbol: str, amount: float) -> Action:
        """Sell a dollar amount of an asset during current increment.
        If symbol is empty str, do nothing. """
        if symbol == '':
            return
        
        self._increment_small_time()
        trade =  self._transactions.sell(
            date=self._date, symbol=symbol, amount=amount)
        self._update_state()
        self.current_actions.append(trade)
        return trade
        
        
    def sell_percent(self, symbol: str, amount: float) -> Action:
        """Sell a percentage of an asset during current increment."""
        self._increment_small_time()
        trade = self._transactions.sell_percent(
            date=self._date, symbol=symbol, amount=amount)
        self._update_state()
        self.current_actions.append(trade)
        return trade


    def sell_delisted(self) -> np.ndarray:
        """Symbols symbols may become delisted. Force sell them.
        
        Returns
        -------
        out : np.ndarray[str]
            Symbols that have been sold due to delisting.
        """
        
        shares = self._transactions.get_asset_shares(self._date)
        
        asset_values = shares.values
        asset_names = shares.index.values
        asset_names = asset_names[asset_values > 0]
        symbols = np.intersect1d(asset_names, self.unlisted_symbols)
        for symbol in symbols:
            self.sell_percent(symbol, amount=1.0)    
        
        if len(symbols) > 0:
            warnings.warn(
                f'The following stocks have been delisted for {self.date}: '
                f'{symbols}')
        return symbols  

        
    # def dataset(self, data: BaseData):
    #     """Register a dataset."""
    #     self._indicators.append(data)
    #     return IndicatorValue(indicators=data, strategy=self)
    

    
    def __notused__indicator(self,
                  func: Callable, 
                  *args,
                  name: str=None, 
                  **kwargs) -> "IndicatorValue":
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
        IndicatorValue
            Object used to retrieve indicator values in `self.next` for
            each stock symbol. For example:
            
            .. code-block :: python
                        
                def init(self):
                    self.my_indicator = self.indicator(func1)
                    
                def next(self):
                    symbol = 'MSFT'
                    value = self.my_indicator(symbol)

        """        
        indicator = Indicators()
        name = indicator.create(func, *args, name=name, **kwargs)
        self._indicators.append(indicator)
        return IndicatorValue(indicators=indicator, strategy=self)
    
    
    def __notused__set_indicator_data(self, stock_data: BaseData):
        """Use to set stock data used for indicator calculation."""
        indicators = self._indicators
        for indicator in indicators:
            indicator.set_stock_data(stock_data)
            
            
    @property
    def days_since_start(self):
        """Days since the beginning of simulation for the current simulation increment."""
        return self._days_since_start
    

class IndicatorManager:
    """Create and retrieve indicator data."""
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self.data_sources = {}
    
    
    def from_basedata(self, d: BaseData, key: str):
        """Create from a dictionary of dataframes
        
        Parameters
        ----------
        d : dict[pandas.DataFrame] or BaseData
            dict of dataframes with date index and indicator columns.
        """
        d = IndicatorBaseData(d, self.strategy)
        self.data_sources[key] = d
        
    
    def from_dataframe_dict(self, d: dict[pd.DataFrame], key: str):
        d = DictData(d)
        return self.from_basedata(d, key)
        
        
    def from_dataframe(self, d: pd.DataFrame, key: str):
        """Create from a DataFrame of date index and column symbols.
        
        Parameters
        ----------
        d : pandas.DataFrame
            Dataframe of date index an column symbols
        """
        d = IndicatorDataFrame(d, name=key, strategy=self.strategy)
        self.data_sources[key] = d
        
    
    def from_func(self, func: Callable, *args, key: str, **kwargs):
        indicator = Indicators()
        stock_data = self.strategy._transactions.stock_data
        indicator.set_stock_data(stock_data)
        
        name = indicator.create(func, *args, name=key, **kwargs)
        d = IndicatorBaseData(indicator, self.strategy)
        self.data_sources[key] = d
        
        
    def keys(self):
        return list(self.data_sources.keys())
    
    
    def __getitem__(self, key: str):
        return self.data_sources[key]
    
    
    def ____old_get_latest(self, date: np.datetime64=None) -> pd.DataFrame:
        """Get all indicators for latest time."""
        series_dict = {}
        source : IndicatorBaseData
        for key in self.data_sources:
            source = self.data_sources[key]
            try: 
                series = source()
                series_dict[key] = series

            except TypeError:
                columns = source.columns
                if len(columns) == 1:
                    series = source[columns[0]]
                    series_dict[key] = series
                else:
                    for column in columns:
                        series = source[column]
                        newname = key + '.' + column
                        series_dict[newname] = series
        df = pd.DataFrame.from_dict(series_dict, orient='index')
        return df
    
    
    def get_latest(self, date: np.datetime64=None) -> pd.DataFrame:
        series_dict = {}
        for key in self.data_sources:
            source = self.data_sources[key]
            data = source()

                
            if hasattr(data, 'columns'):
                columns = data.columns
                cnum = len(columns)
                if cnum == 1:
                    series_dict[key] = data[columns[0]]
                else:
                    for column in columns:
                        name = key + '.' + column
                        series_dict[name] = data[column]
            else:    
                series_dict[key] = data
            
        df = pd.DataFrame.from_dict(series_dict, orient='index')
        return df
    
    
    def __call__(self, date: np.datetime64=None) -> pd.DataFrame:
        return self.get_latest(date)
                    
    
class IndicatorBaseData:
    """Indicator dataframe-dict interface to strategy."""
    def __init__(self, basedata: BaseData, strategy: Strategy):
        self.basedata = basedata
        tables = {}
        for column in self.columns:
            df = basedata.extract_column(column)
            tables[column] = TableData(df)
        self.tables = tables
        self.strategy = strategy
        
        
    @cached_property
    def columns(self):
        key = next(iter(self.basedata.tables))
        table = self.basedata.tables[key]
        return table.columns
    

    
    def _series(self, column: str, date: np.datetime64=None) -> pd.Series:
        if date is None:
            date = self.strategy.date
        return self.tables[column].series_at(date)
    
    def latest(self, date: np.datetime64=None) -> pd.DataFrame:
        d = {}
        for column in self.columns:
            d[column] = self._series(column, date)
        return pd.DataFrame.from_dict(d)
    
    
    def __call__(self, date: np.datetime64=None) -> pd.DataFrame:
        return self.latest(date)
    
    
    
    def __getitem__(self, column: str) -> pd.Series:
        """Retrieve data from specified column and date (latest if None)."""
        return self.latest(column)
        


class IndicatorDataFrame:
    """Indicator dataframe interface to strategy"""
    def __init__(self, df: pd.DataFrame, name: str, strategy: Strategy):
        self.table = TableData(df)
        self.name = name
        self.strategy = strategy
        
    
    def latest(self, date: np.datetime64=None) -> pd.Series:
        if date is None:
            date = self.strategy.date
        return self.table.series_at(date)
    
    
    @cached_property
    def columns(self):
        """Names of data columns."""
        return np.array([self.name])
    
    
    def __call__(self, date: np.datetime64=None) -> pd.Series:
        return self.latest(date)
    
    
    def __getitem__(self, column: str) -> pd.Series:
        """Retrieve data from specified column and date (latest if None)."""
        if column == self.name:
            return self.latest()
        else:
            s = f'Column {column} not available. Only column available is {self.name}.'
            raise KeyError(s)
        


class IndicatorValue:
    """Store reference to indicator, retrieve the indicator data for 
    a given symbol."""    
    def __init__(self,
                 indicators: Indicators,
                 strategy: Strategy):
        self.indicators = indicators
        self.strategy = strategy
        
        
    def array(self, symbol: str):
        date = self.strategy.date
        table = self.indicators.tables[symbol]
        array = table.array_up_to(date)
        return array
    
    
    def dataframe(self, symbol: str):
        date = self.strategy.date
        table = self.indicators.tables[symbol]
        df = table.dataframe_up_to(date)
        return df
    
    
    @cached_property
    def columns(self):
        key = next(iter(self.indicators.tables))
        table = self.indicators.tables[key]
        return table.columns
    
    
    def latest(self, symbol: str):
        """Get latest value for the current Strategy date."""
        table = self.indicators.tables[symbol]
        date = self.strategy.date
        return table.array_at(date)
    
    
    def __call__(self, symbol: str):
        return self.array(symbol)
    
    

# class StrategyState:
#     def __init__(self, strategy: Strategy):
#         self._strategy = strategy
#         self._transactions = strategy._transactions
#         self._indicators = strategy._indicators
        
        
        
#     def _update(self):
#         delete_attr(self, 'asset_values')
#         delete_attr(self, 'stock_data')
#         delete_attr(self, 'existing_symbols')
#         delete_attr(self, 'unlisted_symbols')
#         delete_attr(self, 'equity')    
        
#     def _update_transactions(self):
#         delete_attr(self, 'available_funds')
#         delete_attr(self, 'asset_values')    
    
    
    
#     @property
#     def date(self):
#         """The date of the current simulation increment."""
#         return self._strategy._date
    
    
#     @property
#     def days_since_start(self):
#         """Days since the beginning of simulation for the current simulation increment."""
#         return self._strategy._days_since_start

    
    
#     @cached_property
#     def available_funds(self) -> float:
#         """Available cash for trading during current increment."""
#         return self._transactions.last_available_funds()
    
    
#     @cached_property
#     def asset_values(self) -> pd.Series:
#         """Pandas Series of asset value of each traded symbol for current increment."""
#         return self._transactions.get_asset_values(self._date)
    
    
#     @cached_property
#     def equity(self) -> float:
#         funds = self.available_funds
#         try:
#             assets = self.asset_values.values.sum()
#         except IndexError:
#             assets = 0
#         return funds + assets

    
#     @cached_property
#     def stock_data(self) -> dict[pd.DataFrame]:
#         """dict[DataFrame] : Dataframes for each symbol for the current increment."""
#         return self._indicators.stock_data.filter_dates(end=self.date)  

    
#     @cached_property
#     def existing_symbols(self) -> list[str]:
#         """Symbols which exist at the current date increment."""
#         return self._indicators.stock_data.existing_symbols(self.date)
    
    
#     @cached_property
#     def unlisted_symbols(self) -> list[str]:
#         """Symbols which do not exist at the current date increment."""
#         return self._indicators.stock_data.unlisted_symbols(self.date)
    
    
#     def return_ratios(self):
#         self._transactions.reset
    
    
    
    
    
    
class _IndicatorValue:
    """Store reference to indicator, retrieve the indicator data for 
    a given symbol."""
    def __init__(self, name: str, 
                 indicators: Indicators,
                 strategy: Strategy):
        self.name = name
        self.indicators = indicators
        self.strategy = strategy

        
    @cached_property
    def _name_loc(self):
        symbol = next(iter(self.indicators.tables))
        table = self.indicators.tables[symbol]
        columns = table.columns
        loc = columns.tolist().index(self.name)
        return loc
    
    
    def __call__(self, symbol: str):
        """Retrieve indicator value for given symbol up to current date."""
        return self.array(symbol)
    
    
    def array(self, symbol: str):
        date = self.strategy.date
        table = self.indicators.tables[symbol]
        array = table.array_up_to(date)
        return array[:, self._name_loc]
    
            
    
    def get_last(self, symbol: str, default=np.nan):
        try:
            arr = self.__call__(symbol)
        except NotEnoughDataError:
            return default
        try:
            return arr[-1]
        except IndexError:
            return default
    
        

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
    transactions : Transactions
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
            price_name = DF_ADJ_CLOSE,
            ):

        self.stock_data = stock_data
        self.transactions = Transactions(
            stock_data = stock_data,
            init_funds = cash,
            commission = commission,
            price_name = price_name,
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
        self.reset()
        return
    
    
    def run(self):
        # self.strategy : Strategy
        # self.strategy._indicators.set_stock_data(self.stock_data)
        # self.strategy.init()
        # self.transactions.hold(self.active_days[0])
        self.reset()
        
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
        
        
    def reset(self, time_index=0):
        """Reset backtester."""
        
        date = self.active_days[0]
        
        self.strategy : Strategy
        self.strategy.init()
        # self.strategy.set_indicator_data(self.stock_data)
        self.strategy._set_data(date, 0)
        
        self.transactions.reset()
        self.transactions.hold(date)
        self.time_index = time_index
        

    def step(self):
        """Increment one step of backtest simulation."""
        ii = self.time_index
        active_day = self.active_days[ii]
        self.strategy._set_data(
            date = active_day,
            days_since_start = self._time_int[ii]
            )
        self.strategy.next()
        self.time_index += 1
    
    
    
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
    def asset_values(self) -> pd.DataFrame:
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
            d[symbol] = st.get_share_valuations(dates)
            
        df = pd.DataFrame(d)
        df = df.set_index(self.COLUMN_DATE, drop=True)

        # Get cash and total equity
        df0 : TransactionsDF
        df0 = transactions.dataframe
        cash_a  = df0[self.COLUMN_FUNDS]
        dates_a = df0.index
                
        df[self.COLUMN_FUNDS] = interp_const_after(dates_a, cash_a, dates)
        df[self.COLUMN_EQUITY] = df.sum(axis=1)
        return df
 
    
    @cached_property
    def performance(self) -> pd.DataFrame:
        """Normalized performance of bot vs considered stocks."""
        st : SymbolTransactions
        transactions = self._transactions
        symbols = transactions.get_symbols_list()
        dates = self._active_days
        
        d = {}
        # d[self.COLUMN_DATE] = dates
        
        # Get stock price
        for symbol in symbols:
            st = transactions.get_symbol_transactions(symbol)
            price = st.get_prices(dates)
            d[symbol] = price / price[0]
        
        # Get our equity
        equity = self.asset_values[self.COLUMN_EQUITY]
        equity = equity / equity[0]
        d[self.COLUMN_EQUITY] = equity
        return pd.DataFrame(d)
    
    
    def benchmark(self, symbol: str):
        """Get performance of a buy and hold for the same time frame."""
        dates = self._active_days
        mask = np.array([0, -1])
        dates2 = dates[mask]
        
        transactions = self._transactions
        
        st = SymbolTransactions(
            symbol, 
            stock_data=transactions.stock_data,
            commission=transactions.commission
            )        
        
        prices = st.get_prices(dates2)
        
        return prices[-1] / prices[0]
    
    
    @lru_cache
    def mean_returns(self, window: int):
        series1 = self.performance[self.COLUMN_EQUITY]
        r1 = TrailingIntervals(series1, window_size=window).return_ratio
        return r1
    
    
    @lru_cache
    def benchmark_returns(self, symbol: str, window: int):
        series1 = self.performance[self.COLUMN_EQUITY]
        index = series1.index        
        transactions = self._transactions
        st = SymbolTransactions(
            symbol, 
            stock_data = transactions.stock_data,
            commission = transactions.commission,
            )   
        prices = st.get_prices(index.values)
        series2 = pd.Series(data=prices, index=index)

        r2 = TrailingIntervals(series2, window_size=window).return_ratio
        return r2
    
    
    @lru_cache
    def sharpe_ratio(self, symbol: str, window: int):
        """Calculate Sharpe Ratio"""
        r1 = self.mean_returns(window)
        r2 = self.benchmark_returns(symbol, window)
        delta = r1 - r2
        
        mean = np.nanmean(delta)
        std = np.nanstd(delta)
        return mean / std
        
    
    @cached_property
    def exposure_percentages(self):
        """Ratio of exposure time and net equity for each asset,
        to calculate the percentage of resources (time & money) 
        an asset has used."""
        assets = self.asset_values
        dates = (assets.index
                       .values
                       .astype('datetime64[D]')
                       .astype(float))
        
        
        new = {}
        for name in assets.columns:
            values = assets[name].values
            new[name] = np.trapz(values, dates)
        series = pd.Series(data=new)
        
        ending_equity = series[self.COLUMN_EQUITY]
        
        series = series / ending_equity
        return series
    









