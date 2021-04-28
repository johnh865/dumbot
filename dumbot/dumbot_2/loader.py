"""
This bot seems to work pretty well!!!

trailing stat return_ratio is sometimes better than exp_growth!
Actually Eh only sometimes...

"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb
from typing import Callable
from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData, TableData, Indicators
from backtester.exceptions import NotEnoughDataError

from functools import cached_property
import datetime


import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()




yahoo = YahooData()
symbols = yahoo.get_symbol_names()
rs = np.random.default_rng(0)
rs.shuffle(symbols)

STOCKS = symbols[0:50]
STOCKS.append('SPY')
STOCKS.append('VOO')
# STOCKS.append('GOOG')
# STOCKS.append('TSLA')
STOCKS = np.array(STOCKS)


# def post1(df:pd.DataFrame):
#     series = df[DF_ADJ_CLOSE]
#     ts = TrailingStats(series, 252*5)
#     stat1 = ts.exp_growth
#     return stat1

def post1(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, 252*5)
    stat1 = ts.return_ratio
    return stat1



yahoo.symbols = STOCKS
indicator = Indicators(yahoo)
indicator.create(post1)

df = indicator.get_column_from_all('post1()')
table = TableData(df)
print('Load done')





