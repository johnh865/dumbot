# -*- coding: utf-8 -*-
"""Compare a bunch of different stocks."""


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

symbols = ['MSFT', 'AAPL', 'GOOG'] + names[0 : 5]
date1 = np.datetime64('2005-01-01')
date2 = np.datetime64('2010-08-01')


for symbol in symbols:
    df = y.get_symbol_before(symbol, date2)
    ii = df.index.values >= date1
    df = df.loc[ii]
    close = df[DF_ADJ_CLOSE]
    plt.semilogy(close, label=symbol)


plt.legend()



