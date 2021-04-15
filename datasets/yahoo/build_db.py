# -*- coding: utf-8 -*-
"""Download historical data from yahoo finance."""


import yfinance as yf   
from sqlalchemy import create_engine

from datasets import symbols
from datasets.yahoo.definitions import CONNECTION_PATH, TABLE_SYMBOL_PREFIX
from backtester import utils

SYMBOLS = symbols.ALL


def retrieve_symbols(symbols: list[str]):
    if isinstance(symbols, str):
        symbols = [symbols]
    
    start_date = '1990-01-01'
    engine = create_engine(CONNECTION_PATH, echo=False)
    
    symbol_num = len(symbols)
    for ii, symbol in enumerate(symbols):
        print(f'Downloading {symbol} ({ii} out of {symbol_num})')
        df = yf.download(symbol, start=start_date)
        
        name = TABLE_SYMBOL_PREFIX + symbol
        df.to_sql(name, con=engine, if_exists='replace')
        
    
    
if __name__ == '__main__':
    # pass
    retrieve_symbols(symbols.ALL)