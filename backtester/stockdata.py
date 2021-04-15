import datetime
from functools import lru_cache
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


class _SymbolData:
    """Methods to retrieve symbol data faster."""
    def __init__(self, df: pd.DataFrame):
        df = df.sort_index()
        self.times = df.index
        self.df = df
        self.values = df.values
        self.names = df.columns.values.astype(str)
        
    def dataframe_before(self, date: np.datetime64):
        ii = np.searchsorted(self.times, date)
        return self.df.iloc[0 : ii]
    
    
    def dict_before(self, date: np.datetime64):
        """Pandas dataframes are slow. Get dict[np.ndarray] for ~5x speed-up."""
        ii = np.searchsorted(self.times, date)
        v = self.values[0 : ii].T
        return dict(zip(self.names, v))


    def array_before(self, date: np.datetime64):
        """Pandas dataframes are slow. Get values for ~10x speed-up."""
        ii = np.searchsorted(self.times, date)
        return self.values[0 : ii]
    

class BaseData(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('This method needs to be implemented by user')
        

    @abstractmethod
    def get_symbol_names(self) -> list:
        """Get all available symbols."""
        raise NotImplementedError('This method needs to be implemented by user')
        
    @abstractmethod
    def retrieve_symbol(self, symbol: str) -> pd.DataFrame:
        """Get all available dataframe data for a stock ticker symbol."""
        raise NotImplementedError('This method needs to be implemented by user')
        
        
    @lru_cache(maxsize=500)
    def _get_symbol_data(self, symbol: str) -> _SymbolData:
        """Construct SymbolData."""
        df = self.retrieve_symbol(symbol)
        return _SymbolData(df)
            
        
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
            new[symbol] = self.get_symbol_all(symbol)
        return new
       
    
class BaseStockData(BaseData):        
    @abstractmethod
    def get_trade_dates(self) -> np.ndarray:
        """The tradeable dates in the stock database"""
        raise NotImplementedError('This method needs to be implemented by user')
        
        
    
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
            symbols = read_yahoo_symbol_names()
        self._symbols = symbols
        return
    
    
    def retrieve_symbol(self, symbol: str):
        return read_yahoo_dataframe(symbol)
    
    
    def get_symbol_names(self):
        return self._symbols
    
    
    def get_trade_dates(self):
        return read_yahoo_trade_dates()
    

def _get_indicator_name(func, args, kwargs):
    name = func.__name__
    
    args = [str(a) for a in args]
    
    str_args = ','.join(args)
    name += '(' + str_args
    str_kwargs = [f'{k}={v}' for (k,v) in kwargs.items()]
    str_kwargs = ','.join(str_kwargs)
    name += ', ' + str_kwargs + ')'    
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
                    name2 = name + f'["{key}"]'
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
    



    
    

    