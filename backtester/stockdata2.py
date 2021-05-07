# -*- coding: utf-8 -*-

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





class Symbol(NamedTuple):
    name : str
    

class LazyDict(MutableMapping):
    """Lazy dictionary that applies a function only when the item is retrieved.
    
    Parameters
    ----------
    keys : 
        Dictionary keys
    function : Callable
        Function to apply onto keys as `out = function(key)`
    """
    def __init__(self, keys, function: Callable):
        self.function = function
        
        # functions = [[function]] * len(keys)
        functions = [[function] for _ in keys]
        
        self._func_dict = dict(zip(keys, functions))
        self._dict = {}
        self._func_memory = [function]
        
        
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
        return iter(self._dict)
    
    
    def __len__(self):
        return len(self._dict)
        
    
    def apply(self, func):
        """Apply a function on all keys."""
        self._func_memory.append(func)
        for key in self._func_dict.keys():
            
            self._func_dict[key].append(func)
            
    def map(self, func, deep=True):
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
    """Opposite of LazyDict. Recalculates every time item is retrieved."""
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
        
        
    def dataframe_before(self, date: np.datetime64):
        ii = np.searchsorted(self.times, date)
        return self.df.iloc[0 : ii]
    
    
    def dataframe_after(self, date: np.datetime64):
        ii = np.searchsorted(self.times, date)
        return self.df.iloc[ii :]    
    
    
    def dict_before(self, date: np.datetime64):
        """Pandas dataframes are slow. Get dict[np.ndarray] for ~5x speed-up."""
        ii = np.searchsorted(self.times, date)
        v = self.values[0 : ii].T
        return dict(zip(self.names, v))


    def array_before(self, date: np.datetime64):
        """Pandas dataframes are slow. Get values for ~10x speed-up."""
        ii = np.searchsorted(self.times, date)
        return self.values[0 : ii]
    


    def array_right_before(self, date: np.datetime64):
        """Get slice of values"""
        ii = np.searchsorted(self.times, date)
        return self.values[ii - 1]
    
    
    def exists_at(self, date: np.datetime64) -> bool:
        """Check existence of data for the given time. True if within 
        min and max dates in `self.times`."""
        if self._min_date < date <= self._max_date:
            return True
        return False
    
    
    
    
class BaseData:
    
    
    @abstractmethod
    def symbol_names(self) -> list[str]:
        """Get all available symbols."""
        raise NotImplementedError('This method needs to be implemented by user')    
    
    
    @abstractmethod
    def retrieve(self, symbol: str) -> pd.DataFrame:
        """Get all available dataframe data for a stock ticker symbol."""
        raise NotImplementedError('This method needs to be implemented by user')
        
        
    @cached_property
    def dataframes(self) -> RecalcMapper[pd.DataFrame]:
        """Data dict in form of Pandas DataFrame."""

        def func1(name: str):
            return self.tables[name].dataframe

        r = RecalcMapper(self.symbol_names(), function=func1)
        return r
    
    
    @cached_property
    def tables(self) -> LazyDict[TableData]:
        """Data dict in the form of TableData objects."""
        d = LazyDict(self.symbol_names(), function=self.retrieve)
        d = d.map(TableData, deep=False)
        return d
    
        
    def existing_symbols(self, date: np.datetime64) -> list[str]:
        """Sometimes data will not be available for certain dates. Retrieve
        the symbols that exist for a given time."""
        names = self.symbol_names()
        out = []
        for name in names:
            table = self.tables[name]
            if table.exists_at(date):
                out.append(name)
        return out
    
    
    def unlisted_symbols(self, date: np.datetime64) -> list[str]:
        """Sometimes data will not be available for certain dates. Retrieve
        the symbols that DO NOT exist for a given time."""    
        names = self.symbol_names()
        out = []
        for name in names:
            table = self.tables[name]
            if not table.exists_at(date):
                out.append(name)
        return out
    
    
    def extract_column(self, column: str) -> pd.DataFrame:
        """From all symbols retrieve the specified data column"""
        symbols = self.symbol_names()
        symbol = symbols[0]
        table = self.tables[symbol]
        
        new = []
        for symbol in symbols:
            try:
                table = self.tables[symbol]
                series = table.dataframe[column]
                series.name = symbol
                new.append(series)
            except NotEnoughDataError:
                pass
            
        df = pd.concat(new, axis=1, join='outer',)
        return df
    
    
    def map(self, func: Callable):
        new = copy.copy(self)
        new.tables = self.tables.map(func)
        return new

    
    def filter_before(self, date: np.datetime64) -> 'BaseData':
        """Return data before a certain date. 

        Parameters
        ----------
        date : np.datetime64
            Cutoff date.

        Returns
        -------
        BaseData
        """        
        
        def func(table: TableData):
            return TableData(table.dataframe_before(date), sort=False)
        
        new = copy.copy(self)
        new.tables = self.tables.map(func)
        return new
    
    def filter_after(self, date: np.datetime64) -> 'BaseData':
        """Return data after a certain date. 

        Parameters
        ----------
        date : np.datetime64
            Cutoff date.

        Returns
        -------
        BaseData
        """
        
        def func(table: TableData):
            return TableData(table.dataframe_after(date), sort=False)
        
        new = copy.copy(self)
        new.tables = self.tables.map(func)
        return new    
        
    
    
    
class YahooData(BaseData):
    """Retrieve data from Yahoo online dataset."""
    def __init__(self, symbols=()):
        if len(symbols) == 0:
            symbols = read_yahoo_symbol_names(only_good=True)
        self.symbols = symbols
        return
    
    
    def retrieve(self, symbol: str):
        return read_yahoo_dataframe(symbol)
    
    
    def symbol_names(self):
        return self.symbols
    
    
    def get_trade_dates(self):
        return read_yahoo_trade_dates()
    
            
    
    

    
y = YahooData()
date = np.datetime64('2020-01-01')
y2 = y.filter_before(date)
df = y2.dataframes['MSFT']

