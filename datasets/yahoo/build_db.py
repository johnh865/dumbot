# -*- coding: utf-8 -*-
"""Download historical data from yahoo finance."""
import datetime
import pandas as pd
import numpy as np
import yfinance as yf   
from sqlalchemy import create_engine

from datasets import symbols
from datasets.yahoo.definitions import CONNECTION_PATH, TABLE_SYMBOL_PREFIX
from datasets.yahoo.read import read_yahoo_trade_dates
from datasets.yahoo.build_trade_dates import save_trade_dates
from backtester.stockdata import to_sql, to_parquet


from backtester import utils

SYMBOLS = symbols.ALL

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
        
def download(symbols: list[str], start_date = '1990-01-01'):
    if isinstance(symbols, str):
        symbols = [symbols]
    
    
    engine = create_engine(CONNECTION_PATH, echo=False)
    
    symbol_num = len(symbols)
    for ii, symbol in enumerate(symbols):
        print(f'Downloading {symbol} ({ii} out of {symbol_num})')
        df = yf.download(symbol, start=start_date)
        
        name = TABLE_SYMBOL_PREFIX + symbol
        df.to_sql(name, con=engine, if_exists='replace')
    
        
    
    
def download_to_df_dict(symbols: list[str],
              start_date: str='1990-01-01') -> dict[pd.DataFrame]:
    
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



def update(symbols: list[str]):
    """Update Yfinance database"""
    # engine = create_engine(CONNECTION_PATH, echo=False)
    
    slen = len(symbols)
    chunk_size = 500
    # chunk_num = slen // chunk_size + 1
    trade_dates = read_yahoo_trade_dates()
    last_date = trade_dates[-1]
    today = datetime.date.today()
    if last_date == np.datetime64(today):
        print(f'Already updated to {last_date}.')
        return 
    
    last_date = last_date.astype('datetime64[D]').astype(str)
    client = utils.SQLClient(CONNECTION_PATH)
    
    ii = 0
    for symbols_chunk in chunks(symbols, chunk_size):
        df_dict = download_to_df_dict(symbols_chunk, start_date=last_date)
        for symbol, df in df_dict.items():
            print(f'Saving {symbol} ({ii})')
            name = TABLE_SYMBOL_PREFIX + symbol
            client.append_dataframe(df, name)
            
            ii += 1
            
    save_trade_dates()
    return
    
# def update(symbols: list[str]):
    
#     # slen = len(symbols)
#     # chunk_size = 10
#     # chunk_num = slen // chunk_size + 1
#     trade_dates = read_yahoo_trade_dates()
#     last_date = trade_dates[-1]
#     last_date = last_date.astype('datetime64[D]').astype(str)
    
#     client = utils.SQLClient(CONNECTION_PATH)
#     for symbol in symbols:
#         df = yf.download(symbol, start=last_date)        
#         name = TABLE_SYMBOL_PREFIX + symbol
#         client.append_dataframe(df, name)
    

    
    
if __name__ == '__main__':
    pass
    # download(symbols.ALL)
    # update(symbols.ALL)
    # d = download_to_dfdict(symbols.ALL[0:10])