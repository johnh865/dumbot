# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

from backtester.indicators import TrailingStats
from backtester.smooth import TrailingSavGol

from backtester.stockdata import YahooData, Indicators, TableData
from backtester.definitions import DF_ADJ_CLOSE
from scipy.signal import find_peaks

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

date2 = np.datetime64('2020-03-25')
df = yahoo.get_symbol_all('SPY')
df = yahoo.get_symbol_before('SPY', date2)

close = np.log(df[DF_ADJ_CLOSE])

peaks, _ = find_peaks(close, width=3, distance=10, )
troughs, _ = find_peaks(-close, width=3, distance=10, )
plt.plot(close)

x = close.index[peaks]
y = close.values[peaks]
plt.plot(x, y, '-x')

x = close.index[troughs]
y = close.values[troughs]
plt.plot(x, y, '-g')
plt.scatter(x, y, facecolors='none', edgecolors='g')

