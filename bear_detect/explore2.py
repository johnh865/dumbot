# -*- coding: utf-8 -*-

"""The difference between max gain and max loss seems like a good metric."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester.indicators import TrailingStats
from backtester.stockdata import YahooData, Indicators, TableData
from backtester.definitions import DF_ADJ_CLOSE


yahoo = YahooData()
symbols = yahoo.get_symbol_names()

df = yahoo.get_symbol_all('SPY')
close = df[DF_ADJ_CLOSE]

ts1 = TrailingStats(close, 100)
ts2 = TrailingStats(close, 500)

plt.subplot(3,1,1)
plt.plot(df.index, ts1.exp_reg_value )
plt.plot(df.index, ts2.exp_reg_value )
plt.plot(close)
plt.grid()

plt.subplot(3,1,2)
plt.plot(df.index, ts1.max_loss, label='max_loss')
plt.plot(df.index, ts1.max_gain, label='max_gain')
plt.grid()
plt.legend()

plt.subplot(3,1,3)
metric = ts1.max_gain - ts1.max_loss

plt.plot(df.index, metric, label='gain-loss')
plt.axhline(y=0, color='k')
plt.grid()
plt.legend()