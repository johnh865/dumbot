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
    """Ratio of current price to peak price."""
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





def trough_volume2(x: np.ndarray, decay=.95):
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
        current_max = current_max * decay
        
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
def _sub_func1(x, start_locs, end_locs):
    num = len(start_locs)
    xlen = len(x)
    
    out_areas = np.zeros(xlen)
    out_start_locs = []
    out_end_locs = []
    out_areas_final = []
    
    for jj in range(num):
        start = start_locs[jj]
        end = end_locs[jj]
        out = trough_volume(x[start : end])
        
        out_areas[start : end] = out[0]
        
        out_start_locs.append(out[1] + start)
        out_end_locs.append(out[2] + start)
        out_areas_final.append(out[3])
    return out_areas, out_start_locs, out_end_locs, out_areas_final

def _subtrough_volume(x: np.ndarray, start_locs, end_locs):    
    
    out_areas, out_start_locs, out_end_locs, out_areas_final = (
        _sub_func1(x, start_locs, end_locs))
    
    out_start_locs = np.concatenate(out_start_locs)
    out_end_locs = np.concatenate(out_end_locs)
    out_areas_final = np.concatenate(out_areas_final)
    return out_areas, out_start_locs, out_end_locs, out_areas_final


@njit
def trough_locate(x: np.ndarray, start_locs, end_locs):
    loc_num = len(start_locs)
    min_locs = np.zeros(loc_num, dtype=np.int64)
    
    for ii in range(loc_num):
        start = start_locs[ii]
        end = end_locs[ii]
        min_locs[ii] = x[start : end].argmin() + start
    return min_locs


        
    
def trough_finder(x: np.ndarray, p=98):
    xa, starts, ends, areas = trough_volume(x)
    sig_area = np.percentile(areas, p)
    
    sig_locs = areas > sig_area    
    starts = starts[sig_locs]
    ends = ends[sig_locs]    
    
    pxa, pstart, pend, pa = _subtrough_volume(-x, starts, ends)
    psig_area = np.percentile(pa, p)    
    sig_locs2 = pa > sig_area
    
    
    return starts, ends, pstart


@dataclass
class TroughData:
    series: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    areas: np.ndarray
        
        
class TroughFinder:
    def __init__(self, x):
        self.x = x
    
        
    @cached_property
    def peak(self):
        return self._get_peaks(slice(None))
    
    
    @cached_property
    def trough(self):
        return self._get_troughs(self.peak.starts, 
                                 self.peak.ends,
                                 slice(None))
        
    
    @cached_property
    def _trough_volume(self):
        xa, starts, ends, areas = trough_volume(x)
        return xa, starts, ends, areas
    
    
    def _get_peaks(self, mask):
        xa, starts, ends, areas = self._trough_volume
        starts = starts[mask]
        ends = ends[mask]
        areas = areas[mask]
        return TroughData(series=xa, starts=starts, ends=ends, areas=areas)
    
    
    def _get_troughs(self, starts, ends, submask):
        x = self.x
        pxa, pstart, pend, pa = _subtrough_volume(-x, starts, ends)
        pstart = pstart[submask]
        pend = pend[submask]
        pa = pa[submask]
        return TroughData(series=pxa, starts=pstart, ends=pend, areas=pa)
    
    
    def copy(self):
        new = TroughFinder(self.x.copy())
        new._trough_volume = self._trough_volume
        return new

    
    def filter_length(self, time: int):
        """Filter troughs/peaks by time length."""
        ends = self._trough_volume[2]
        starts = self._trough_volume[1]
        tdelta = ends - starts
        mask = tdelta > time

        new = self.copy()
        peak = self._get_peaks(mask)

        pxa, pstart, pend, pa = _subtrough_volume(-x, peak.starts, peak.ends)
        tdelta2 = pend - pstart
        submask = tdelta2 > time
        pstart = pstart[submask]
        pend = pend[submask]
        pa = pa[submask]
        trough = TroughData(series=pxa, starts=pstart, ends=pend, areas=pa)
        
        new.peak = peak
        new.trough = trough
        return new
    
    
    def filter_area(self, area: float):
        mask = self._trough_volume[3] > area
        new = self.copy()
        peak = self._get_peaks(mask)
        
        pxa, pstart, pend, pa = _subtrough_volume(-x, peak.starts, peak.ends)
        submask = pa > area
        pstart = pstart[submask]
        pend = pend[submask]
        pa = pa[submask]
        trough = TroughData(series=pxa, starts=pstart, ends=pend, areas=pa)
        
        new.peak = peak
        new.trough = trough
        return new        
        
        
        
class TroughAssess:
    def __init__(self, series: pd.Series):
        t = TroughFinder(series.values).filter_area(0.5)
        self.trough_finder = t
        self.index = series.index
        
        
        
    def _get_last_locs(self, peaks):
        
        plen = len(peaks)
        tlen = len(self.index)
        
        iarr = np.arange(tlen)
        ii = np.searchsorted(peaks, iarr) - 1
        
        out_bound_locs = ii < 0
        ii = np.maximum(ii, 0)
        out = peaks[ii]
        out[out_bound_locs] = 0
        return out

    @cached_property
    def last_peak_locs(self):
        """Index locations of last detected peak."""
        peaks = self.trough_finder.peak.starts
        return self._get_last_locs(peaks)
    
    
    @cached_property
    def last_trough_locs(self):
        """Index locations of last detected trough."""
        troughs = self.trough_finder.trough.starts
        return self._get_last_locs(troughs)
    
    
    @cached_property
    def last_maxima_locs(self):
        p = self.last_peak_locs
        t = self.last_trough_locs
        return np.maximum(p, t)
        
    @cached_property
    def time_from_trough(self):
        t = self.last_trough_locs
        n = np.arange(len(t))
        return n - t

        
        
        

x = np.log(close.values)

# Get trough locations
t = TroughFinder(x)
# t = t.filter_length(30)
# delta = t.peak.ends - t.peak.starts
# t = t.filter_area(1)

date1 = np.datetime64('1995-01-01')
# date2 = np.datetime64('2006-01-01')
date2 = None

# plt.subplot(2,2,1)
plt.semilogy(close)

plt.plot(close.iloc[t.peak.starts], 'o', label='start')
plt.plot(close.iloc[t.trough.starts], 'x', label='min')
# plt.plot(close.iloc[t.peak.ends], 'x', label='end')
plt.legend()

plt.grid(which='both')
plt.xlim(date1, date2)


tt = TroughAssess(close)

