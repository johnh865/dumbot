"""Explore different window sizes and growth."""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from backtester.stockdata import YahooData
from backtester.definitions import DF_ADJ_CLOSE
from backtester.indicators import TrailingStats

y = YahooData()
start = np.datetime64('2003-01-01')
end = np.datetime64('2005-01-01')
y = y.filter_dates(start=start, end=end)


s = y.dataframes['SPY'][DF_ADJ_CLOSE]

window = 100
windows = [100, ]
for window in windows:
    ts = TrailingStats(s, window)
    loss = ts.max_loss
    # loss99 = np.percentile(loss, 99)
    
    plt.subplot(2,1,1)
    
    
    plt.plot(ts.times, ts.max_loss, label='max loss')
    plt.plot(ts.times, ts.max_gain, label='max gain')
    delta = ts.max_gain - ts.max_loss
    plt.plot(ts.times, delta, label='delta')
    
max_loss_mean = np.mean(ts.max_loss)
max_loss_p = np.percentile(ts.max_loss, 95)
plt.axhline(0.0)
plt.legend()
plt.subplot(2,1,2)
plt.semilogy(s[ts.times])