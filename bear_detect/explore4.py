# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

from backtester.indicators import TrailingStats
from backtester.smooth import TrailingSavGol

from backtester.stockdata import YahooData, Indicators, TableData
from backtester.definitions import DF_ADJ_CLOSE


def break_mask(mask):
    diff = np.abs(np.diff(mask))
    break_points = np.where(diff > 0)[0] + 1
    bnum = len(break_points)
    
    true_mask = []
    false_mask = []
    for ii in range(bnum - 1):
                
        j1 = break_points[ii]
        j2 = break_points[ii + 1]
        imask = np.arange(j1, j2)
        if mask[j1] == True:
            true_mask.append(imask)
        else:
            false_mask.append(imask)
    return true_mask, false_mask


yahoo = YahooData()
symbols = yahoo.get_symbol_names()

df = yahoo.get_symbol_all('SPY')
close = df[DF_ADJ_CLOSE]

# t = np.linspace(0, 10, 1000)
# y = np.sin(t/1)+3

# close = pd.Series(data=y, index=t)


ts = TrailingStats(close, 50)
m1 = ts.exp_growth > 0
m2 = ts.return_ratio > 0
m3 = ts.max_gain

close1 = close.iloc[~m1]
close2 = close.iloc[~m2]
plt.plot(close, 'k')

plt.plot(close1, 'o', ms=2)
plt.plot(close2, 'o', ms=1)


# tmasks, fmasks = break_mask(m1)
# for tmask, fmask in zip(tmasks, fmasks):
#     x = close.index.values[tmask]
#     y = close.values[tmask]
#     plt.plot(x, y, 'r')
    
#     x = close.index.values[fmask]
#     y = close.values[fmask]
#     plt.plot(x, y, 'b')

ax = plt.gca()
# ax.set_yscale('log')
