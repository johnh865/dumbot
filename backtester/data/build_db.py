# -*- coding: utf-8 -*-
"""Download historical data from yahoo finance."""


import yfinance as yf   
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy import create_engine
from dumbot.definitions import CONNECTION_PATH
from dumbot import utils
# stock_data = yf.download(stock_symbol,'2016-01-01','2021-03-01') 


from dumbot.data import symbols
SYMBOLS = symbols.ALL


# Base = declarative_base()
# class StockData(Base):
#     __tablename__ = 'stockdata'
#     id = Column(String, primary_key=True)
#     adj_close = Column(Float)


def retrieve_symbols():
    start_date = '1990-01-01'
    engine = create_engine(CONNECTION_PATH, echo=False)
    
    symbol_num = len(SYMBOLS)
    for ii, symbol in enumerate(SYMBOLS):
        print(f'Downloading {symbol} ({ii} out of {symbol_num})')
        df = yf.download(symbol, start=start_date)
        df.to_sql(symbol, con=engine, if_exists='append')
        

def drop_table(name):
    utils.drop_table(name, CONNECTION_PATH, echo=False)
    
    
if __name__ == '__main__':
    retrieve_symbols()