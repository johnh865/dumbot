# -*- coding: utf-8 -*-

"""Measure statistics of short term growth. Attempt to catch abnormalities
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import backtester

from backtester.indicators import TrailingStats
from backtester.analysis import BuySell, avg_future_growth
from backtester.stockdata import YahooData
from backtester.definitions import DF_ADJ_CLOSE
from backtester import utils
from backtester.smooth import SmoothOptimize, TrailingSavGol


y = YahooData()
names = y.get_symbol_names()
np.random.seed(7)
np.random.shuffle(names)

symbol = 'MSFT'
date1 = np.datetime64('2005-01-01')
date2 = np.datetime64('2018-08-01')


df = y.get_symbol_before(symbol, date2)
ii = df.index.values >= date1
df = df.loc[ii]
close = df[DF_ADJ_CLOSE]

ts = TrailingStats(close, window_size=30)
growth = ts.exp_growth

plt.subplot(3,1,1)
plt.hist(growth, bins=40)

# Calculate statistics and cumulative probability of growth
growth1 = growth.copy()
growth1[np.isnan(growth)] = 0
growth_hist, edges = np.histogram(growth1, bins=100,)
growth_cum = np.cumsum(growth_hist)
percentile = growth_cum / growth_cum[-1]

plt.subplot(3,1,2)
plt.plot(edges[0:-1], percentile)


# Convert growth to probability
percentile1 = np.interp(growth1 ,edges[0:-1], percentile)

ax = plt.subplot(3,1,3)
plt.scatter(close.index, close, c=percentile1, s=5)

p01_locs = percentile1 <= .02
plt.plot(close.index[p01_locs], close[p01_locs], 'rx', mfc=None)


ax.set_yscale('log')
plt.colorbar()

