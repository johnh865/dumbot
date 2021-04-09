import datetime
import math
from functools import lru_cache, cached_property
from abc import abstractmethod, ABCMeta
from typing import Callable

from sqlalchemy import create_engine
import pandas as pd
import numpy as np

from backtest.data.symbols import ALL
from backtest.definitions import (
    CONNECTION_PATH, 
    DF_DATE, DF_ADJ_CLOSE, DF_SMOOTH_CHANGE, DF_SMOOTH_CLOSE,
    DF_TRUE_CHANGE,
    TABLE_ALL_TRADE_DATES,
    )


engine = create_engine(CONNECTION_PATH, echo=False)

@lru_cache(maxsize=100)
def read_yahoo_dataframe(symbol: str):
    """Read all available stock symbol Yahoo data."""
    dataframe = pd.read_sql(symbol, engine).set_index(DF_DATE, drop=True)
    return dataframe
	

class BaseData(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
            
    
    @abstractmethod
    def get_symbol_all(self, symbol:str) -> pd.DataFrame:
        """Get all available dataframe data for a stock ticker symbol"""
        pass


    @abstractmethod
    def get_symbol_names(self) -> list:
        """Get all available symbols."""
        pass
    
    
    def get_symbol_before(self, symbol: str, date: datetime.datetime) -> pd.DataFrame:
        """Retrieve data before a date."""
        date = np.datetime64(date)
        df = self.get_symbol_all(symbol)
        return df[df.index < date]
    
    
    def get_symbols_before(self, date: datetime.datetime) -> dict:
        out = {}
        symbols = self.get_symbol_names()
        for symbol in symbols:
            out[symbol] = self.get_symbol_before(symbol, date)
        return out
    
    
    def get_before(self, symbol: str, name: str, date: datetime.datetime):
        """Return values for a given symbol and indicator name"""
        df = self.get_symbol_before(symbol=symbol, date=date)
        return df[name]
    
    
       
    
class StockData(BaseData):
    
    @abstractmethod
    def get_trade_dates(self) -> np.ndarray:
        """The tradeable dates in the stock database"""
        pass
        
    
    
           
    
class YahooData(StockData):
    def __init__(self, symbols=()):
        if len(symbols) == 0:
            symbols = ALL
        self._symbols = symbols
        return
    
    
    def get_symbol_all(self, symbol: str):
        return read_yahoo_dataframe(symbol)
    
    
    def get_symbol_names(self):
        return self._symbols
    
    
    def get_trade_dates(self):
        df = pd.read_sql(TABLE_ALL_TRADE_DATES, engine)
        return df[DF_DATE].values
    

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
    def __init__(self, stock_data: StockData=None):
        self.stock_data = stock_data
        self._indicators = {}
        self._class_indicators = {}
        self._locked = False
        
        
    def get_symbol_names(self):
        return self.stock_data.get_symbol_names()


    def set_stock_data(self, stock_data: StockData):
        """Set stock data."""
        self.stock_data = stock_data

        
    def create(self, func: Callable, *args, name=None, **kwargs) -> str:
        """Create an indicator from function of form:
            value = func(*args, df=df, **kwargs)
        
        Parameters
        ----------
        func : Callable
            Function to convert to indicator. It must accept a dataframe 
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
            
            
        
        
    @lru_cache   
    def get_symbol_all(self, symbol: str):
        """Calculate indicators for the given symbol."""
        self._locked = True
        df = self.stock_data.get_symbol_all(symbol)
        df2 = df[[]].copy()
               
        for name, idata in self._indicators.items():
            (func, args, kwargs) = idata
            value = func(*args, df=df, **kwargs)
            if type(value) == tuple:
                for jj, v in enumerate(value):
                    name2 = name + f'[{jj}]'
                    df2[name2] = v
                    
            elif issubclass(dict, type(value)):
                for key, v in value.items():
                    name2 = name + f'["{key}"]'
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
    
    

    

        
    
    

    
    