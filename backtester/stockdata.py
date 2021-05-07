# import pdb
import datetime
from functools import lru_cache, cached_property
from abc import abstractmethod, ABCMeta
from typing import Callable
from typing import Union

import pandas as pd
import numpy as np

from datasets.symbols import ALL
from datasets.yahoo import (
    read_yahoo_dataframe, 
    read_yahoo_symbol_names,
    read_yahoo_trade_dates,
    )
from backtester.exceptions import DataError, NotEnoughDataError

class TableData:
    """Methods to retrieve symbol data faster."""
    def __init__(self, df: pd.DataFrame):
        df = df.sort_index()
        self.times = df.index
        self.df = df
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
    
    
    def dict_before(self, date: np.datetime64):
        """Pandas dataframes are slow. Get dict[np.ndarray] for ~5x speed-up."""
        ii = np.searchsorted(self.times, date)
        v = self.values[0 : ii].T
        return dict(zip(self.columns, v))


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
    

class BaseData(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('This method needs to be implemented by user')
        

    @abstractmethod
    def get_symbol_names(self) -> list[str]:
        """Get all available symbols."""
        raise NotImplementedError('This method needs to be implemented by user')
        
        
        
    @abstractmethod
    def retrieve_symbol(self, symbol: str) -> pd.DataFrame:
        """Get all available dataframe data for a stock ticker symbol."""
        raise NotImplementedError('This method needs to be implemented by user')

        
    def existing_symbols(self, date: np.datetime64) -> list[str]:
        """Sometimes data will not be available for certain dates. Retrieve
        the symbols that exist for a given time."""
        names = self.get_symbol_names()
        out = []
        for name in names:
            table = self._get_symbol_data(name)
            if table.exists_at(date):
                out.append(name)
        return out
    
    
    def unlisted_symbols(self, date: np.datetime64) -> list[str]:
        """Sometimes data will not be available for certain dates. Retrieve
        the symbols that DO NOT exist for a given time."""    
        names = self.get_symbol_names()
        out = []
        for name in names:
            table = self._get_symbol_data(name)
            if not table.exists_at(date):
                out.append(name)
        return out



        
        
    @lru_cache(maxsize=1000)
    def _get_symbol_data(self, symbol: str) -> TableData:
        """Construct SymbolData."""
        df = self.retrieve_symbol(symbol)
        return TableData(df)
            
        
    def get_symbol_all(self, symbol:str) -> pd.DataFrame:
        """Get all available dataframe data for a stock ticker symbol"""
        return self._get_symbol_data(symbol).df
        
    
    def get_symbol_before(self, symbol: str, date: np.datetime64) -> pd.DataFrame:
        """Retrieve data before a date."""
        sd = self._get_symbol_data(symbol)
        return sd.dataframe_before(date)
        
    
    def get_symbols_before(self, date: datetime.datetime) -> dict:
        """Get dataframes for multiple symbols before date."""
        out = {}
        symbols = self.get_symbol_names()
        for symbol in symbols:
            out[symbol] = self.get_symbol_before(symbol, date)
        return out

    
    def __getitem__(self, symbol: str):
        return self._get_symbol_data(symbol).df
    
    

            
    
    
    
    # def get_before(self, symbol: str, name: str, date: np.datetime64):
    #     """Return values for a given symbol and indicator name"""
    #     df = self.get_symbol_before(symbol=symbol, date=date)
    #     return df[name]
    
    
    @lru_cache(maxsize=500)
    def get_dict_before(self, symbol: str, date: np.datetime64) -> np.ndarray:
        """Fast way to return dataframe values before a date."""
        symbol_data = self._get_symbol_data(symbol)
        return symbol_data.dict_before(date)
    
    
    def get_all(self) -> dict[pd.DataFrame]:
        """Convert all data to dict[pd.DataFrame]."""
        symbols = self.get_symbol_names()
        new = {}
        for symbol in symbols:
            try:
                new[symbol] = self.get_symbol_all(symbol)
            except NotEnoughDataError:
                pass
        return new
    
    
    def get_column_from_all(self, column: str) -> pd.DataFrame:
        """From all symbols retrieve the specified data column"""
        symbols = self.get_symbol_names()
        symbol = symbols[0]
        table = self._get_symbol_data(symbol)
        columns = table.columns
        # cloc = np.where(columns == column)[0][0]
        
        new = []
        for symbol in symbols:
            try:
                table = self._get_symbol_data(symbol)
                # values = table.values[:, cloc]
                series = table.df[column]
                series.name = symbol
                new.append(series)
            except NotEnoughDataError:
                pass
            
        df = pd.concat(new, axis=1, join='outer',)
        return df
       
    
class BaseStockData(BaseData):        
    @abstractmethod
    def get_trade_dates(self) -> np.ndarray:
        """The tradeable dates in the stock database"""
        raise NotImplementedError('This method needs to be implemented by user')
        
        
    # @cached_property
    # def _existence_df(self):
    #     """Get trade dates where symbol exists or not."""
    #     dates = self.get_trade_dates()
    #     symbols = self.get_symbol_names()
        
    #     new = np.zeros((len(dates), len(symbols)), dtype=bool)
        
    #     for jj, symbol in enumerate(symbols):
    #         df = self.retrieve_symbol(symbol)
    #         sdates = df.index.values
    #         ii = np.searchsorted(dates, sdates)
            
    #         assert np.all(dates[ii] == sdates)
    #         new[ii, jj] = True
            
    #     return pd.DataFrame(data=new, index=dates, columns=symbols)
    

            
            
        
    
class DictData(BaseData):
    def __init__(self, d: Union[dict[pd.DataFrame], BaseData]):
        
        if hasattr(d, 'get_all'):
            self.dict = d.get_all()
        else:
            self.dict = d
        
        
    def retrieve_symbol(self, symbol: str):
        return self.dict[symbol]
    
    
    def get_symbol_names(self):
        return list(self.dict.keys())
            
           
    
class YahooData(BaseStockData):
    """Retrieve data from Yahoo online dataset."""
    def __init__(self, symbols=()):
        if len(symbols) == 0:
            symbols = read_yahoo_symbol_names(only_good=True)
        self.symbols = symbols
        return
    
    
    def retrieve_symbol(self, symbol: str):
        return read_yahoo_dataframe(symbol)
    
    
    def get_symbol_names(self):
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
        """


        """
        self.stock_data = stock_data
        self._indicators = {}
        self._class_indicators = {}
        self._locked = False
        
        
        
    def get_symbol_names(self):
        return self.stock_data.get_symbol_names()


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
        
        
    def create_obj(self, cls: object, *args, name=None, **kwargs):
        """Create an indicator from an object of form:
            
            > obj1 = cls(*args, df=df, **kwargs)
            > indicators = obj1.indicators 
            > value1 = obj1.attr1
            > value2 = obj1.attr2
            > value3 = obj1.attr3
            > ....
        """
        if self._locked:
            raise Exception('You cannot create more indicators '
                             'if self.get_symbol_all has been called')
        if name is None:
            name = _get_indicator_name(cls, args, kwargs)
        self._class_indicators[name] = (cls, args, kwargs)
            
            
    def retrieve_symbol(self, symbol: str):
        """Calculate indicators for the given symbol."""
        self._locked = True
        df = self.stock_data.get_symbol_all(symbol)
        df2 = df[[]].copy()
        df_len = len(df.index)
        
        def check_len(value, name_):
            vlen = len(value)
            if vlen != df_len:
                raise ValueError(
                    f'Indicator {name_} length {vlen} does not match '
                    f'stock_data length {df_len} for current increment '
                    f'for symbol {symbol}.')
               
        for name, idata in self._indicators.items():
            (func, args, kwargs) = idata
            try:
                value = func(df, *args, **kwargs)
            except TypeError:
                try:
                    value = func(*args, df=df, **kwargs)
                except TypeError:
                    value = func(*args, **kwargs)
                
                
            if type(value) == tuple:
                for jj, v in enumerate(value):
                    name2 = name + f'[{jj}]'
                    check_len(v, name2)
                    df2[name2] = v
                    
            elif issubclass(dict, type(value)):
                for key, v in value.items():
                    name2 = name + f"['{key}']"
                    check_len(v, name2)
                    df2[name2] = v
            else:
                df2[name] = value
            
            
        for name, idata in self._class_indicators.items():
            (cls, args, kwargs) = idata
            obj = cls(*args, df=df, **kwargs)
            obj_indicators = obj.indicators
            for name2 in obj_indicators:
                value = getattr(obj, name2)
                newname = name + '.' + name2
                df2[newname] = value
                
        return df2
    



    
    

    