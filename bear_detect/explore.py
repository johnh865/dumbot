# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester.indicators import TrailingStats
from backtester.stockdata import YahooData, Indicators, TableData
from backtester.definitions import DF_ADJ_CLOSE


yahoo = YahooData()
symbols = yahoo.get_symbol_names()
rs = np.random.default_rng(0)
rs.shuffle(symbols)

STOCKS = symbols[0:200]
STOCKS.append('SPY')
STOCKS.append('VOO')
# STOCKS.append('GOOG')
# STOCKS.append('TSLA')
STOCKS = np.array(STOCKS)


def post2(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, 10)
    stat1 = ts.exp_growth
    
    mean = np.nanmean(stat1)
    std = np.nanstd(stat1)    
    return (stat1 - mean) / std


def post1(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, 15)
    stat1 = ts.return_ratio
    
    mean = np.nanmean(stat1)
    std = np.nanstd(stat1)
    return (stat1 - mean) / std



yahoo.symbols = STOCKS
indicator = Indicators(yahoo)
indicator.create(post1)

df = indicator.get_column_from_all('post1()')

spy = yahoo.get_symbol_all('SPY')[DF_ADJ_CLOSE]

table = TableData(df)
# %%
ii = df.values < -1.5
stocknum = ii.shape[-1]
isum = np.sum(ii, axis=1)

imean = np.nanmean(df.values, axis=1)
# date1 = np.datetime64('2008-01-01')
# date2 = np.datetime64('2009-01-01')
date1 = None
date2 = None
plt.subplot(2,2,1)
plt.plot(df.index, imean)
plt.grid()
plt.xlim(date1, date2)


plt.subplot(2,2,2)
plt.plot(df.index, isum/stocknum)
plt.grid()
plt.xlim(date1, date2)

ax = plt.subplot(2, 2, 3)
plt.plot(spy)
ax.set_yscale('log')
plt.grid(which='both')
plt.xlim(date1, date2)
