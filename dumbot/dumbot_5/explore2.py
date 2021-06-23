"""Explore different window sizes and growth."""

# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from backtester.stockdata import YahooData
from backtester.definitions import DF_ADJ_CLOSE
from backtester.indicators import TrailingStats

y = YahooData()
s = y.dataframes['SPY'][DF_ADJ_CLOSE]

windows = [5, 10, 15, 20, 30, 40]
d = {}
for window in windows:
    ts = TrailingStats(s, window)
    d[window] = pd.Series(ts.exp_growth, index=ts.times)
    
df = pd.concat(d, axis=1, join='outer',)

plt.subplot(2,1,1)
for key in df:
    plt.plot(df[key], label=key)
plt.legend()
plt.subplot(2,1,2)
plt.semilogy(s)