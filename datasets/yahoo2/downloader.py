# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import numpy as np
import pdb
import os 
import logging 


from backtester.utils import ParquetClient
from datasets.yahoo.definitions import (
    DF_DATE,
    TABLE_ALL_TRADE_DATES,
    TABLE_SYMBOL_PREFIX,
    TABLE_GOOD_SYMBOL_DATA,
    PARQUET_PATH
    )

import yfinance as yf   
from datasets.symbols import ALL as ALL_SYMBOLS
from datasets.yahoo.clean import is_clean_symbol


logger = logging.getLogger(__name__)

def download_df_dict(
        symbols: list[str],
        start_date: str = '1990-01-01',
        end_date: str=None,
        ) -> dict[str, pd.DataFrame]:
    
    dframes = yf.download(symbols, 
                          start=start_date,
                          end=end_date,
                          group_by='ticker',
                          )
    dframes.index = dframes.index.values.astype('datetime64[ns]')
    dframes.index.name = 'Date'
    new = {}
    assert dframes.columns.nlevels == 2
    
    keys = dframes.columns.get_level_values(0)
    keys = np.unique(keys)
    for key in keys:
        df = dframes[key].dropna()
        if df.size > 0:
            new[key] = df
    return new


def download_df_yr_dict(
        symbols: list[str],
        start_yr: int=1990,
        end_yr: int=None,
        ) -> dict[int, dict[str, pd.DataFrame]]:
    
    start = str(datetime.date(start_yr, 1, 1))
    
    if end_yr is None:
        end = datetime.datetime.now()
        end_yr = end.year
    else:
        day1 = datetime.timedelta(days=1)
        end = datetime.date(end_yr + 1, 1, 1) - day1
        
    dict_i = download_df_dict(symbols, start, end)
    # pdb.set_trace()
    
    # Construct yr_dict to store outputs in nested dictionary
    years1 = range(start_yr, end_yr + 1)
    yr_dict = {}
    for year in years1:
        yr_dict[year] = {}
    
    
    for symbol, df in dict_i.items():
        years = df.index.year
        years_unique = years.unique()
        
        for year in years_unique:
            df2 = df.loc[years == year]
            yr_dict[year][symbol] = df2
    return yr_dict
    
    
    
    
    # if end_yr is None:
    #     end_yr = datetime.datetime.now().year
    
    # years = range(start_yr, end_yr + 1)
    # day1 = datetime.timedelta(days=1)
    
    # starts = [datetime.date(yr, 1, 1) for yr in years]
    # ends = [start - day1 for start in starts[1:]] 
    # ends.append(None)
    
    # dict1 = {}
    # for ii, year in enumerate(years):
        
    #     start = str(starts[ii])
    #     end = str(ends[ii])
    #     dict_i = download_df_dict(symbols, start, end)
    #     dict1[year] = dict_i
        
    # return dict1
        
    
    
    
    
    
class TradeDates:
    """Get trading dates available in Yahoo database."""
    
    def __init__(self, path: str, start_date: str='1990-01-01'):
        self.start_date = start_date
        self.client = ParquetClient(path)

        
    def _build_series(self):
        tickers = ['SPY', 'GOOG', 'DIS']
        df = yf.download(tickers, start=self.start_date)
        return df.index
        
        
    def update(self):
        series = self._build_series()
        df = series.to_frame()
        logger.info('Updating trade dates')
        self.client.save_dataframe(df, TABLE_ALL_TRADE_DATES)
        return series.values
        
        
    def _read(self):
        df =  self.client.read_dataframe(TABLE_ALL_TRADE_DATES)
        return df.values[:,0]
    
    
    def get(self):
        try: 
            return self._read()
        except FileNotFoundError:
            return self.update()
        



class YahooClient:
    """Download and read Yahoo stock data."""
    def __init__(self, path: str, symbols=()):
        
        self.client = ParquetClient(path)
        self._symbols_desired = symbols
        self.trade_dates = TradeDates(path)
        # self._names = self.client.get_table_names()
        self.symbols = self.get_symbols()
        
        
        
    def get_symbols(self):
        names = self.client.get_table_names()
        
        symbols = []
        for name in names:
            if name.startswith('symbol-'):
                symbols.append(name[7:])
        return symbols
    
    
    def add_symbols(self, symbols: list[str], start_date=None, end_date=None):
        """Add a symbol to parquet data."""
        dict_i = download_df_dict(
            symbols,
            start_date = start_date,
            end_date = end_date
            )
        
        for symbol, df in dict_i.items():
            self.client.save_dataframe(df, 'symbol-' + symbol)
        self.symbols = list(dict_i.keys())
            
            
            
        
        
    def init_data(self, start_date=None, end_date=None):
        symbols = self._symbols_desired
        logger.info(
            'Initializing data starting %s, ending %s',
            start_date, 
            end_date)
        
        return self.add_symbols(symbols, start_date, end_date)
        
            
        
    def read(self, symbol: str) -> pd.DataFrame:
        """Read stock dataframe."""
        
        return self.client.read_dataframe('symbol-' + symbol)
    
    
    def update(self, end_date=None):
        
        # Get last trading date
        self.trade_dates.update()
        today = self.trade_dates.get()[-1].astype('datetime64[D]')
        
        # date = datetime.date.today()
        if len(self.symbols) == 0:
            self.init_data(end_date=end_date)
            return
        
        
        if 'SPY' in self.symbols:
            test_symbol = 'SPY'
        else:
            test_symbol = self.symbols[0]
        
        date_last = self.read(test_symbol).index[-1] 
          
        date1 = date_last + pd.DateOffset(days=1)
        date1 = str(date1.date())        
        
        # Do not update if last detected date is already updated. 
        if str(date_last.date()) == str(today):
            return
        
        symbols = self.symbols
        symbols.extend(self._symbols_desired)
        symbols = set(symbols)

        dicti = download_df_dict(
            
            symbols,
            start_date = str(date1),
            end_date = end_date
            
            )
        for symbol, df in dicti.items():
            self.client.append_dataframe(df, 'symbol-' + symbol)
        self.symbols = self.get_symbols()
        
        
            
                
                
    # def read_symbol(self, symbol: str):
    #     self.client.
                
    # def get(self, symbol: str, years: list[str]=None):
    #     if years is None:
    #     self.client.read_dataframe(name)
                
            


# d = test_download_df_yr_dict()
# d = test_download_yesterday()

