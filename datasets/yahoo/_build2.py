# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import numpy as np


from backtester.stockdata import ParquetData
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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
  
        
      
def download(
        symbols: list[str],
        start_date: str = '1990-01-01'):
    
    if isinstance(symbols, str):
        symbols = [symbols]
    
    new = {}
    symbol_num = len(symbols)
    for ii, symbol in enumerate(symbols):
        print(f'Downloading {symbol} ({ii} out of {symbol_num})')
        df = yf.download(symbol, start=start_date)
        new[symbol] = df
    return new
    
    
def download_to_df_dict(
        symbols: list[str],
        start_date: str='1990-01-01'
        ) -> dict[pd.DataFrame]:
    
    dframes = yf.download(symbols, 
                          start=start_date,
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

        
class YahooDownloader:
    def __init__(self, symbols=()):
        self.client = ParquetClient(PARQUET_PATH)
        if len(symbols) == 0:
            symbols = ALL_SYMBOLS
        self.symbols = symbols
        self.trade_dates = TradeDates()
        
        
    def update(self):
        self.update_symbols(self.symbols)
        self.update_clean_list()
        

    def update_symbols(self, symbols: list[str]):
        """Update Yfinance database"""
        # engine = create_engine(CONNECTION_PATH, echo=False)
        chunk_size = 500
        update_flag = False
        
        try:
            last_date = self.trade_dates.read()[-1]
        except FileNotFoundError:
            update_flag = True
            last_date = -1
            
        trade_date_new = self.trade_dates.update()[-1]
        if trade_date_new != last_date:
            update_flag = True
        
        if not update_flag:
            print(f'Already updated to {trade_date_new}.')
            return 
        
        if last_date == -1:
            last_date = '1990-01-01'
        else:
            last_date = last_date.astype('datetime64[D]').astype(str)
        client = self.client
        
        ii = 0
        for symbols_chunk in chunks(symbols, chunk_size):
            df_dict = download_to_df_dict(
                symbols_chunk,
                start_date = last_date
                )
            
            for symbol, df in df_dict.items():
                print(f'Saving {symbol} ({ii})')
                
                name = TABLE_SYMBOL_PREFIX + symbol
                
                if client.table_exists(name):
                    
                    client.append_dataframe(df, name)

                else:
                    client.save_dataframe(df, name)
                
                ii += 1
                
        return
    
    
    def reader(self):
        p = ParquetData(self.client.dir_path)
        return p
    
    
    def read_symbol(self, name):
        name = TABLE_SYMBOL_PREFIX + name
        return self.client.read_dataframe(name)
    
    
    def update_clean_list(self):
        good_list = self.get_clean_symbols()
        df = pd.Series(good_list, name='symbol').to_frame()
        self.client.save_dataframe(df, TABLE_GOOD_SYMBOL_DATA)
        
    
    def get_symbols(self) -> list[str]:
        tables = self.client.get_table_names()
        new = []
        for table_name in tables:
            if table_name.startswith(TABLE_SYMBOL_PREFIX):
                new.append(table_name.replace(TABLE_SYMBOL_PREFIX, ''))
        return new

    
    def get_clean_symbols(self) -> list[str]:
        symbols = self.get_symbols()
        new = []
        for symbol in symbols:
            name = TABLE_SYMBOL_PREFIX + symbol
            df = self.client.read_dataframe(name)
            if is_clean_symbol(df, symbol):
                new.append(symbol)
        return new
    

    
class TradeDates:
    def __init__(self, start_date: str='1990-01-01'):
        self.start_date = start_date
        self.client = ParquetClient(PARQUET_PATH)

        
    def build(self):
        tickers = ['SPY', 'GOOG', 'DIS']
        df = yf.download(tickers, start=self.start_date)
        return df.index
        
        
    def update(self):
        series = self.build()
        df = series.to_frame()
        self.client.save_dataframe(df, TABLE_ALL_TRADE_DATES)
        return series.values
        
        
    def read(self):
        df =  self.client.read_dataframe(TABLE_ALL_TRADE_DATES)
        return df.values
    
    
    def get(self):
        try: 
            return self.read()
        except FileNotFoundError:
            return self.update()
        


    
class YahooData(ParquetData):
    """Get data from directory of parquet files."""

    def __init__(self, symbols=()):
        directory = PARQUET_PATH
        prefix = TABLE_SYMBOL_PREFIX
        super().__init__(directory=directory, symbols=symbols, prefix=prefix)



def test_trade_dates():
    t = TradeDates()
    df = t.update()
    df = t.read()
    
    
def test_downloader():
    d = YahooDownloader(symbols=('DIS', 'SPY', 'GOOG', 'TSLA'))
    d.update()
            
    
if __name__ == '__main__':
    d = YahooDownloader()
    d.update()

    
    

            
    
