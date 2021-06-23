# -*- coding: utf-8 -*-
from datasets import symbols
from datasets.yahoo.build_db import download
from datasets.yahoo.build_trade_dates import save_trade_dates
from datasets.yahoo.clean import save_good_table


def build():
    ALL = symbols.ALL
    download(ALL)
    save_trade_dates()
    save_good_table()
    
    
if __name__ == '__main__':
    build()