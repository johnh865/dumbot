# -*- coding: utf-8 -*-
"""A bunch of classes and functions used to store and retrieve stock data."""

import pdb
import datetime
import copy

from functools import lru_cache, cached_property
from abc import abstractmethod, ABCMeta
from typing import Callable
from typing import Union, NamedTuple
from collections.abc import MutableMapping, Mapping

import pandas as pd
import numpy as np

from datasets.symbols import ALL
from datasets.yahoo import (
    read_yahoo_dataframe, 
    read_yahoo_symbol_names,
    read_yahoo_trade_dates,
    )
from backtester.exceptions import DataError, NotEnoughDataError
from backtester.utils import SQLClient, ParquetClient


TABLE_PREFIX = 'symbol-'
SYMBOL_TABLE = 'good-symbol-list'


class DataFrameStocks(pd.DataFrame):
    """DataFrame with dates as index and stock symbol names as columns."""
    pass


class DataFrameStock(pd.DataFrame):
    """DataFrame with dates as index and data for single stock as columns."""
    pass





class LazyMap(MutableMapping):
    """Lazy dictionary that applies a function only when the item is retrieved.
    
    Parameters
    ----------
    keys : 
        Dictionary keys
    function : Callable or list[Callable]
        Function to apply onto keys as `out = function(key)`
    """
    def __init__(self, keys, function: Callable, values=None):
        self.function = function
        
        if hasattr(function, '__iter__'):
            functions = [function for _ in keys]
        else:
            functions = [[function] for _ in keys]
        
        self._func_dict = dict(zip(keys, functions))
        self._dict = {}
        self._func_memory = [function]
        
        if values is not None:
            self._dict = dict(zip(keys, values))
        
        
    def __getitem__(self, key):
    
        functions = self._func_dict[key]
        if functions:
            try:
                out = self._dict[key]
            except KeyError:
                out = key

            for func in functions:
                out = func(out)
                
            self._dict[key] = out
            self._func_dict[key] = []
            return out            
            
        else:
            return self._dict[key]
        
        
    def __setitem__(self, key, value):
        self._dict[key] = value
        self._func_dict[key] = []
    
        
    def __delitem__(self, key):
        del self._dict[key]
        del self._func_dict[key]
        
        
    def __iter__(self):
        return iter(self._func_dict)
    
    
    def __len__(self):
        return len(self._func_dict)
        
    
    def apply(self, func: Callable):
        """Apply a function on all keys."""
        
        if hasattr(func, '__iter__'):
            self._func_memory.extend(func)
            for key in self._func_dict.keys():
                self._func_dict[key].extend(func)
                
        else:
            self._func_memory.append(func)
            for key in self._func_dict.keys():
                self._func_dict[key].append(func)
                
                
    def map(self, func, deep=True):
        """Create copy and map a function on all keys."""
        d = self.copy(deep=deep)
        
        d.apply(func)
        return d
    
            
    def add(self, key):
        self._func_dict[key] = self._func_memory
        
        
    def copy(self, deep=True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
        


class RecalcMapper(Mapping):
    """Opposite of LazyMap. Recalculates every time item is retrieved."""
    def __init__(self, keys, function: Callable):
        self.function = function        
        self._func_memory = [function]
        self._keys = keys
        

    def __getitem__(self, key):
        functions = self._func_memory
        out = key
        for func in functions:
            out = func(out)
        return out
    
    
    def __len__(self):
        return len(self._keys)
    
    
    def __iter__(self):
        return iter(self._keys)
    
    
    def as_lazy(self):
        return LazyMap(self._keys, self._func_memory)
    
        
    
class TableData:
    """Methods to retrieve data faster from a dataframe."""
    def __init__(self, df: pd.DataFrame, sort=True):
        
        if sort:
            df = df.sort_index()
        self.times = df.index
        self.df = df
        self.dataframe = df
        self.values = df.values
        try:
            self.columns = df.columns.values.astype(str)
        except AttributeError:
            self.columns = [df.name]
            
        self._min_date = np.min(self.times)
        self._max_date = np.max(self.times)
        
        
    def date_index(self, date: np.datetime64):
        """Get index location of data for a given date."""
        return np.searchsorted(self.times, date)
    
    
    def iloc(self, index):
        """Set index locations."""
        df = self.iloc_dataframe(index)
        return TableData(df, sort=False)
    
    
    def iloc_dataframe(self, index):
        """Set index locations for dataframe"""
        return self.df.iloc[index]
    
    
    def iloc_dict(self, index):
        """Set index locations for dict"""
        v = self.values[index].T
        return dict(zip(self.columns, v))
    
    
    def iloc_array(self, index):
        """Set index locations for array"""
        return self.values[index]

        
    def dataframe_up_to(self, date: np.datetime64):
        ii = np.searchsorted(self.times, date)
        return self.df.iloc[0 : ii + 1]
    
    
    def dict_up_to(self, date: np.datetime64):
        """Pandas dataframes are slow. Get dict[np.ndarray] for ~5x speed-up."""
        ii = np.searchsorted(self.times, date)
        v = self.values[0 : ii + 1].T
        return dict(zip(self.columns, v))


    def array_up_to(self, date: np.datetime64):
        """Pandas dataframes are slow. Get values for ~10x speed-up."""
        ii = np.searchsorted(self.times, date)
        return self.values[0 : ii + 1]
    
    def array_at(self, date: np.datetime64):
        ii = np.searchsorted(self.times, date)
        return self.values[ii]
    
    
    def series_at(self, date: np.datetime64):
        ii = np.searchsorted(self.times, date)
        return self.df.iloc[ii]
    
    
    # def dataframe_after(self, date: np.datetime64):
    #     ii = np.searchsorted(self.times, date)
    #     return self.df.iloc[ii :]    
    
    
    # def array_right_before(self, date: np.datetime64):
    #     """Get slice of values"""
    #     ii = np.searchsorted(self.times, date)
    #     return self.values[ii - 1]
    
    
    def exists_at(self, date: np.datetime64) -> bool:
        """Check existence of data for the given time. True if within 
        min and max dates in `self.times`."""
        if self._min_date < date <= self._max_date:
            return True
        return False
    
    
    
    
class BaseData(metaclass=ABCMeta):
    """Base class for creating stock data."""
    
    
    @abstractmethod
    def retrieve_symbol_names(self) -> list[str]:
        """Method to retrieve all available symbols."""
        raise NotImplementedError('This method needs to be implemented by user')    
    
    
    @abstractmethod
    def retrieve(self, symbol: str) -> pd.DataFrame:
        """Get all available dataframe data for a stock ticker symbol."""
        raise NotImplementedError('This method needs to be implemented by user')
        
        
    @cached_property
    def symbol_names(self) -> list[str]:
        """Use this to set which symbol names to consider."""
        return self.retrieve_symbol_names()
        
        
    @cached_property
    def dataframes(self) -> RecalcMapper[pd.DataFrame]:
        """Data dict in form of Pandas DataFrame."""

        def func1(name: str):
            return self.tables[name].dataframe

        r = RecalcMapper(self.symbol_names, function=func1)
        return r
    
    
    @cached_property
    def tables(self) -> LazyMap[TableData]:
        """Data dict in the form of TableData objects."""
        d = LazyMap(self.symbol_names, function=self.retrieve)
        d = d.map(TableData, deep=False)
        return d
    
        
    def existing_symbols(self, date: np.datetime64) -> list[str]:
        """Sometimes data will not be available for certain dates. Retrieve
        the symbols that exist for a given time."""
        names = self.symbol_names
        out = []
        for name in names:
            table = self.tables[name]
            if table.exists_at(date):
                out.append(name)
        return out
    
    
    def unlisted_symbols(self, date: np.datetime64) -> list[str]:
        """Sometimes data will not be available for certain dates. Retrieve
        the symbols that DO NOT exist for a given time."""    
        names = self.symbol_names
        out = []
        for name in names:
            table = self.tables[name]
            if not table.exists_at(date):
                out.append(name)
        return out
    
    
    def extract_column(self, column: str) -> DataFrameStocks:
        """From all symbols retrieve the specified data column"""
        symbols = self.symbol_names
        symbol = symbols[0]
        table = self.tables[symbol]
        
        new = []
        for symbol in symbols:
            try:
                table = self.tables[symbol]
                try:
                    series = table.dataframe[column]
                    series.name = symbol
                    new.append(series)
                except KeyError:
                    pass
            except NotEnoughDataError:
                pass
            
        df = pd.concat(new, axis=1, join='outer',)
        return df
    
    
    def map(self, func: Callable):
        new = copy.deepcopy(self)
        new.tables = self.tables.map(func)
        return new
    
    
    def filter_dates(self, 
                     start: np.datetime64=None,
                     end: np.datetime64=None) -> 'BaseData':
        """Return data within certain dates. 

        Parameters
        ----------
        start : np.datetime64, optional
            Starting date. The default is None to start at index 0
        end : np.datetime64, optional
            Ending date. The default is None to end at index -1

        Returns
        -------
        BaseData
        """
        def func(table: TableData):
            if start is not None:
                i1 = table.date_index(start)
            else:
                i1 = 0
                
            if end is not None:
                i2 = table.date_index(end) + 1
            else:
                i2 = None
            mask = slice(i1, i2)
            return table.iloc(mask)
        
        new = copy.deepcopy(self)
        new.tables = self.tables.map(func)
        return new

    
    # def filter_before(self, date: np.datetime64) -> 'BaseData':
    #     """Return data before a certain date. 

    #     Parameters
    #     ----------
    #     date : np.datetime64
    #         Cutoff date.

    #     Returns
    #     -------
    #     BaseData
    #     """        
        
    #     def func(table: TableData):
    #         return TableData(table.dataframe_before(date), sort=False)
        
    #     new = copy.deepcopy(self)
    #     new.tables = self.tables.map(func)
    #     return new
    
    # def filter_after(self, date: np.datetime64) -> 'BaseData':
    #     """Return data after a certain date. 

    #     Parameters
    #     ----------
    #     date : np.datetime64
    #         Cutoff date.

    #     Returns
    #     -------
    #     BaseData
    #     """
        
    #     def func(table: TableData):
    #         return TableData(table.dataframe_after(date), sort=False)
        
    #     new = copy.deepcopy(self)
    #     new.tables = self.tables.map(func)
    #     return new    
    
    
    def filter_symbols(self, symbols:list[str]):
        self.symbol_names = symbols
        keep = set(symbols)
        old = set(self.symbol_names)
        diff = old.difference(keep)
        
        new = copy.deepcopy(self)
        for key in diff:
            del new.tables[key]
        return new
        
        

class MapData(BaseData):
    def  __init__(self, keys, function: Callable, values=None):
        self._lazy_map = LazyMap(keys, function, values)
        
        
    def retrieve(self, symbol: str):
        return self._lazy_map[symbol]
    
    
    def retrieve_symbol_names(self):
        return list(self._lazy_map.keys())
    
    
class DictData(BaseData):
    def __init__(self, d: dict):
        self.dict = d
    
        
    def retrieve(self, symbol: str):
        return self.dict[symbol]
    
    
    def retrieve_symbol_names(self):
        return list(self.dict.keys())

    
class SQLData(BaseData):
    """Get data from SQL.
    
    Parameters
    ----------
    url : str
        Database name or URL.
    kwargs : TYPE, optional
        optional arguments to pass into create_engine. The default is None.
    symbols : TYPE, optional
        Symbols whitelist. The default is () to consider all symbols.
    prefix : str, optional
        Table prefix. The default is TABLE_PREFIX.
    """    
    def __init__(self, url, kwargs=None, symbols=(), prefix: str=TABLE_PREFIX):

        if kwargs is None:
            kwargs = {}
        self.client = SQLClient(url, **kwargs)
        self.prefix = prefix

        if len(symbols) == 0:
            symbols = self.retrieve_symbol_names()
        self.symbols = symbols
        
        
    def retrieve(self, symbol: str):
        name = self.prefix + symbol
        return self.client.read_dataframe(name)
    
    
    def retrieve_symbol_names(self):
        name = SYMBOL_TABLE
        df = self.client.read_dataframe(name)
        return list(df['symbol'].values)
    
    
class ParquetData(SQLData):
    """Get data from directory of parquet files."""
    def __init__(self, directory: str, symbols=(), prefix: str=TABLE_PREFIX):
        self.client = ParquetClient(directory)
        self.prefix = prefix

        if len(symbols) == 0:
            symbols = self.retrieve_symbol_names()
        self.symbols = symbols
    
    
class YahooData(BaseData):
    """Retrieve data from Yahoo online dataset."""
    def __init__(self, symbols=()):
        if len(symbols) == 0:
            symbols = read_yahoo_symbol_names(only_good=True)
        self.symbols = symbols
        return
    
    
    def retrieve(self, symbol: str):
        return read_yahoo_dataframe(symbol)
    
    
    def retrieve_symbol_names(self):
        return self.symbols
    
    
    def get_trade_dates(self):
        return read_yahoo_trade_dates()

    
def _get_indicator_name(func, args, kwargs):
    name = func.__name__
    
    args = [str(a) for a in args]
    
    str_args = ','.join(args)
    name += '(' + str_args
    str_kwargs = [f'{k}={v}' for (k,v) in kwargs.items()]
    str_kwargs = ','.join(str_kwargs)
    if len(str_kwargs) > 0:
        name += ', ' + str_kwargs
    name += ')'    
    return name
        
    
class Indicators(BaseData):
    """Create indicators for stock data `StockData` for use in backtesting. 

    Parameters
    ----------
    stock_data : StockData, optional
        `StockData` for which to create indicators. The default is None.
        Eventually to generate indicators, you must assign self.stock_data.
    """
    
    def __init__(self, stock_data: BaseData=None):

        self.stock_data = stock_data
        self._indicators = {}
        self._locked = False
        
        
        
    def retrieve_symbol_names(self):
        return self.stock_data.symbol_names


    def set_stock_data(self, stock_data: BaseData):
        """Set stock data."""
        self.stock_data = stock_data

        
    def create(self, func: Callable, *args, name=None, **kwargs) -> str:
        """Create an indicator from function of form:
            value = func(*args, df=df, **kwargs)
        
        Parameters
        ----------
        func : Callable
            Function to convert to indicator. It must accept a dataframe or dict 
            as a keyword argument `df`. 
        *args, **kwargs :
            Additional arguments for func.
        name : str
            Name of indicator (Optional).
            
        Returns
        -------
        out : str
            New name of indicator
            
        """
        if self._locked:
            raise Exception('You cannot create more indicators '
                             'if self.get_symbol_all has been called')
        if not callable(func):
            raise TypeError('func must be Callable')
            
        if name is None:
            name = _get_indicator_name(func, args, kwargs)

        indicator1 = (func, args, kwargs)
        self._indicators[name] = indicator1
        return name
    
            
    def retrieve(self, symbol: str):
        """Calculate indicators for the given symbol."""
        self._locked = True
        df = self.stock_data.dataframes[symbol]
        # df2 = df[[]].copy()
        # df_len = len(df.index)
        df2 = None
        
        # def check_len(value, name_):
        #     vlen = len(value)
        #     if vlen != df_len:
        #         raise ValueError(
        #             f'Indicator {name_} length {vlen} does not match '
        #             f'stock_data length {df_len} for current increment '
        #             f'for symbol {symbol}.')    
        
                   
        for name, idata in self._indicators.items():
            (func, args, kwargs) = idata
            try:
                value = func(*args, df=df, **kwargs)                
            except TypeError:
                value = func(df, *args, **kwargs)
                
            # check_len(value, name)
            try:
                df2 = self._retrieve_set_values(df2, value, name, symbol)
            except ValueError as exception:
                s = str(exception)
                s += f', symbol={symbol}, column={name}'
                s += f', for function {func}'
                raise ValueError(s)
            # df_len = 
            
        # for name, idata in self._class_indicators.items():
        #     (cls, args, kwargs) = idata
        #     obj = cls(*args, df=df, **kwargs)
        #     obj_indicators = obj.indicators
        #     for name2 in obj_indicators:
        #         value = getattr(obj, name2)
        #         newname = name + '.' + name2
        #         df2[newname] = value
                
        return df2
    
            
    def _retrieve_set_values(self, 
                             df: pd.DataFrame,
                             value, 
                             name: str,
                             symbol: str):
        if df is None:
            
            # Use index of value if Pandas index can be found.
            if (
                isinstance(value, pd.DataFrame) or 
                isinstance(value, pd.Series)
                ):
                df = pd.DataFrame(index=value.index)
            
            # Use index of original dataframes otherwise
            else:
                key = next(iter(self.stock_data.dataframes))
                df0 = self.stock_data.dataframes[symbol]
                df = pd.DataFrame(index=df0.index)
                
        
        if type(value) == tuple:
            for jj, v in enumerate(value):
                name2 = name + f'[{jj}]'
                df[name2] = v
                
        elif issubclass(dict, type(value)):
            for key, v in value.items():
                name2 = name + f"['{key}']"
                df[name2] = v            
        else:
            df[name] = value
        return df

    
def to_sql(connection, data: dict,
           symbol_prefix='symbol-',
           symbol_table='good-symbol-list'):
    
    # TABLE_GOOD_SYMBOL_DATA = 'good-symbol-list'
    # TABLE_SYMBOL_PREFIX = 'symbol-'
    client = SQLClient(connection)

    names = data.keys()
    df = pd.Series(names, name='symbol').to_frame()
    with client.connect():
        client.save_dataframe(df, symbol_table)

        for name in names:
            df = data[name]
            table_name = symbol_prefix + name
            client.save_dataframe(df, table_name)

            
def to_parquet(directory, data: dict[pd.DataFrame], 
           symbol_prefix='symbol-',
           symbol_table='good-symbol-list'):
    """Save dict of DataFrames to a parquet directory."""
    client = ParquetClient(directory)
    names = data.keys()
    df = pd.Series(names, name='symbol').to_frame()
    
    if client.table_exists(symbol_table):
        client.append_dataframe(df, symbol_table)
    else:
        client.save_dataframe(df, symbol_table)
        
    for name in names:
        df = data[name]
        table_name = symbol_prefix + name
        
        try:
            client.save_dataframe(df, table_name)
            
        # Convert pd.Series to pd.DataFrame if needed. 
        except AttributeError:
            df = pd.DataFrame(df)
            client.save_dataframe(df, table_name)
    
    



