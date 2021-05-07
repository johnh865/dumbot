# -*- coding: utf-8 -*-

from functools import cached_property

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

from backtester.indicators import TrailingStats
from backtester.smooth import TrailingSavGol

from backtester.stockdata import YahooData, Indicators, TableData
from backtester.definitions import DF_ADJ_CLOSE
from backtester.utils import interp_const_after
from dataclasses import dataclass



@njit
def trough_volume(x: np.ndarray):
    current_max = x[0]
    current_max_loc = 0
    xlen = len(x)
    areas = np.zeros(xlen)
    
    # Record some information about each trough
    start_locs = np.zeros(xlen, dtype=np.int64)
    end_locs = np.zeros(xlen, dtype=np.int64)
    # min_locs = np.zeros(xlen, dtype=np.int64)
    areas_final = np.zeros(xlen)
    
    area = 0
    jj = 0
    for ii in range(xlen):
        xi = x[ii]
        
        # Price is rising
        if xi > current_max:
            start_locs[jj] = current_max_loc
            end_locs[jj] = ii
            areas_final[jj] = area
            jj += 1
            area = 0
            current_max_loc = ii
            current_max = xi
        # Trough detected
        else:
            area += (current_max - xi)
        
        areas[ii] = area    
        
    # If there's are left over record it for the final trough. 
    if area > 0:
        start_locs[jj] = current_max_loc
        end_locs[jj] = ii
        areas_final[jj] = area
        jj += 1
        
    start_locs = start_locs[0 : jj]
    end_locs = end_locs[0 : jj]
    areas_final = areas_final[0 : jj]
    return areas, start_locs, end_locs, areas_final

@njit
def peak_ratio(x: np.ndarray):
    current_max = x[0]
    current_max_loc = 0
    xlen = len(x)
    
    # Record some information about each trough
    start_locs = np.zeros(xlen, dtype=np.int64)
    end_locs = np.zeros(xlen, dtype=np.int64)
    pratios = np.zeros(xlen)
    
    area = 0
    jj = 0
    for ii in range(xlen):
        xi = x[ii]
        
        # Price is rising
        if xi > current_max:
            start_locs[jj] = current_max_loc
            end_locs[jj] = ii
            jj += 1
            area = 0
            current_max_loc = ii
            current_max = xi
            pratios[ii] = 0
        # Trough detected
        else:
            area += (current_max - xi)
            pratios[ii] = (current_max - xi) / current_max        
    return pratios





yahoo = YahooData()
symbols = yahoo.get_symbol_names()

df = yahoo.get_symbol_all('GOOG')
close = df[DF_ADJ_CLOSE]
xlen = len(close)
iarr = np.arange(xlen)
# a, starts, ends, af = trough_volume(close.values)
# a2, starts2, ends2, af2 = trough_volume(-close.values[::-1])
# starts2 = iarr[::-1][starts2]

ratio = peak_ratio(close.values)
series = pd.Series(data=ratio, index=close.index)
ts = TrailingStats(series, 12)
slope = ts.linear_regression[0]



plt.subplot(2,2,1)
plt.plot(close)
plt.grid()

plt.subplot(2,2,2)
plt.plot(close.index, ratio, '.-', ms=2)
plt.grid()

plt.subplot(2,2,3)
plt.plot(close.index, slope)
plt.axhline(color='k')
plt.grid()



