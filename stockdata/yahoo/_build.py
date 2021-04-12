# -*- coding: utf-8 -*-
from stockdata import symbols
from stockdata.yahoo.build_db import retrieve_symbols
from stockdata.yahoo.build_trade_dates import build_trade_dates


def build():
    ALL = symbols.ALL
    retrieve_symbols(ALL)
    build_trade_dates()
    
    
if __name__ == '__main__':
    build()