# -*- coding: utf-8 -*-

"""Measure statistics of short term growth. Attempt to catch abnormalities
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import backtester

from backtester.indicators import TrailingStats, trailing_percentiles
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
date1 = np.datetime64('200-01-01')
date2 = np.datetime64('2010-01-01')


df = y.get_symbol_before(symbol, date2)
ii = df.index.values >= date1
df = df.loc[ii]
close = df[DF_ADJ_CLOSE]

ts = TrailingStats(close, window_size=40)
growth = ts.exp_growth
max_loss = ts.max_loss
plt.subplot(2,2,1)
plt.hist(max_loss, bins=40)

max_loss1 = max_loss[~np.isnan(max_loss)]
max_loss_percentiles = np.percentile(max_loss1, np.arange(100))
plt.subplot(2,2,2)
plt.plot(max_loss_percentiles)

# Calculate statistics and cumulative probability of growth

ax = plt.subplot(2,2,3)
tp = trailing_percentiles(max_loss, window=500)
plt.scatter(close.index, close, c=tp, s=5)
ax.set_yscale('log')

ax = plt.subplot(2,2,4)
tp_grad = np.gradient(tp)
plt.plot(close.index, tp)



# max_gain = ts.max_gain
# tp = trailing_percentiles(max_gain, window=500)
# plt.scatter(close.index, close, c=tp, s=5)
# ax.set_yscale('log')