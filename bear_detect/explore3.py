# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

from backtester.indicators import TrailingStats
from backtester.smooth import TrailingSavGol

from backtester.stockdata import YahooData, Indicators, TableData
from backtester.definitions import DF_ADJ_CLOSE


yahoo = YahooData()
symbols = yahoo.get_symbol_names()

df = yahoo.get_symbol_all('SPY')
close = df[DF_ADJ_CLOSE]

@njit
def trough_volume(x: np.ndarray):
    current_max = x[0]
    current_max_loc = 0
    xlen = len(x)
    areas = np.zeros(xlen)
    
    # Record some information about each trough
    start_locs = np.zeros(xlen, dtype=np.int64)
    end_locs = np.zeros(xlen, dtype=np.int64)
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


def subpeak_volume(x: np.ndarray, start_locs, end_locs, min_size=2):
    times = end_locs - start_locs
    
    # Filter out  small toughs. 
    ii = times > min_size
    start_locs2 = start_locs[ii]
    end_locs2 = end_locs[ii]
    
    num = len(start_locs2)
    xlen = len(x)
    
    out_areas = np.zeros(xlen)
    out_start_locs = []
    out_end_locs = []
    out_areas_final = []
    
    for jj in range(num):
        start = start_locs2[jj]
        end = end_locs2[jj]
        out = trough_volume(x[start : end])
        
        out_areas[start : end] = out[0]
        
        out_start_locs.append(out[1] + start)
        out_end_locs.append(out[2] + start)
        out_areas_final.append(out[3])
        
    out_start_locs = np.concatenate(out_start_locs)
    out_end_locs = np.concatenate(out_end_locs)
    out_areas_final = np.concatenate(out_areas_final)
    return out_areas, out_start_locs, out_end_locs, out_areas_final


def peak_finder(x :np.ndarray, p=95):
    
    xpeak = -x
    xa, starts, ends, areas = trough_volume(x)
    sig_area = np.percentile(areas, p)
    sig_locs = areas > sig_area

    starts = starts[sig_locs]
    ends = ends[sig_locs]
    areas = areas[sig_locs]    
    
    
    while np.count_nonzero(sig_locs) > 0:
    

        xa, starts, ends, areas = subpeak_volume(x, starts, ends,)
        
        sig_locs = areas > sig_area
        
        starts = starts[sig_locs]
        ends = ends[sig_locs]
        areas = areas[sig_locs]        

class Trough:
    def __init__(self, x: np.ndarray):
        self.x = x
        
        self.areas, self.start_locs, self.end_locs, self.areas_final = (
            trough_volume(x)
            )
        return
    
    def peaks(self):
        pass

        
        
        
        

x = np.log(close.values)

# Get trough locations
a, start_locs, end_locs, areas_final = trough_volume(x)
pmax = areas_final > np.percentile(areas_final, 95)
start_locs = start_locs[pmax]
end_locs = end_locs[pmax]
areas_final = areas_final[pmax]

# Get peak locations
a_p, start_p, end_p, af_p = subpeak_volume(-x, start_locs, end_locs)
pmax = af_p > np.percentile(af_p, 95)
start_p = start_p[pmax]
end_p = end_p[pmax]
af_p = af_p[pmax]



# Get locs of biggest trough
imax = np.argmax(areas_final)
x_imax = close.values[start_locs[imax] : end_locs[imax]]
t_imax = close.index.values[start_locs[imax] :end_locs[imax]]

ts = TrailingStats(close, 600)
tsg = TrailingSavGol(close, 601, polyorder=3)
growth = ts.exp_growth

date1 = np.datetime64('2002-01-01')
# date2 = np.datetime64('2006-01-01')
date2 = None

plt.subplot(2,2,1)
plt.semilogy(close)

plt.plot(close.iloc[start_locs], 'o', label='start')
# plt.plot(close.iloc[end_locs], 'o', label='end')
plt.plot(close.iloc[start_p], 'x', label='start')
# plt.plot(close.iloc[end_p], 'x', label='end')
plt.legend()

plt.semilogy(t_imax, x_imax)
plt.grid(which='both')
plt.xlim(date1, date2)

plt.subplot(2,2,2)
plt.semilogy(close.index, a)
plt.semilogy(close.index, a_p)
plt.grid(which='both')
plt.xlim(date1, date2)
plt.ylim(1, None)

plt.subplot(2,2,3)
plt.plot(close.index, a)
plt.plot(close.index, a_p)
plt.plot(close.index, a_p - a)
plt.grid(which='both')
plt.xlim(date1, date2)

plt.subplot(2,2,4)

diffa = np.diff(a_p - a)
plt.plot(close.index[0:-1], diffa)
plt.axhline(color='k')
plt.grid(which='both')
plt.xlim(date1, date2)



