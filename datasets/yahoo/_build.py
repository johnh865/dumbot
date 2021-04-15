# -*- coding: utf-8 -*-
from datasets import symbols
from datasets.yahoo.build_db import retrieve_symbols
from datasets.yahoo.build_trade_dates import build_trade_dates


def build():
    ALL = symbols.ALL
    retrieve_symbols(ALL)
    build_trade_dates()
    
    
if __name__ == '__main__':
    build()