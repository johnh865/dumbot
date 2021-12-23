# -*- coding: utf-8 -*-
from functools import cached_property

import pandas as pd
import numpy as np
from numba import njit

class ROI:
    def __init__(self, series: pd.Series, interval: int):
        self.series = series
        self.values = self.series.values
        self.interval = interval
        
        
    @cached_property
    def _time_days_int_noskip(self):
        """ndarray ; Time past in days starting from initial time."""
        tdelta = self.series.index - self.series.index[0]
        try:
            tdelta = tdelta.astype('timedelta64[D]')
        except TypeError:
            pass
        return np.asarray(tdelta)
    
    
    @cached_property
    def index_interval_starts(self):
        """Index location breaks for ROI return calculation intervals."""
        times = self._time_days_int_noskip
        tmax = times[-1]
        
        # Make sure last interval is complete. First interval will be off. 
        times2 = tmax - times
        
        interval_counts = times2 / self.interval 
        imax = interval_counts.max() // 1
        interval_counts = imax - interval_counts
        
        return np.searchsorted(interval_counts, np.arange(imax))
    
    
    @cached_property
    def _closes(self):
        index = self.index_interval_starts
        closes = self.values[index] 
        return closes
    

    @cached_property
    def _roi_interval(self):
        """Return on investment ratios."""
        c2 = self._closes[1:]
        c1 = self._closes[0 : -1]
        return (c2 - c1) / c1
    
        
    @cached_property
    def _rof_interval(self):
        """Return vs final price ratios."""
        c2 = self._closes[1:]
        c1 = self._closes[0 : -1]
        return (c2 - c1) / c2   
    
    
    @cached_property
    def times(self):
        return self.series.index.values[self.index_interval_starts]
    
    
    @cached_property
    def times_end(self):
        return self.times[1:]
    
    
    @cached_property
    def times_start(self):
        return self.times[0 : -1]
    
    
    @cached_property
    def times_delta(self):
        """ndarray[float] : Change in time in days from start to end of interval."""
        delta = self.times_end - self.times_start
        delta = delta.astype('timedelta64[D]')
        return delta.astype(float)
    
    
    @cached_property
    def annualized(self):
        """ndarray[float] : Annualized rate of return."""
        r = self._roi_interval
        annualized = (1 + r)**(365/self.times_delta) - 1
        return annualized
    
    
    @cached_property
    def annualized_adjusted(self):
        """ndarray[float] : Special adjusted annualized rate of return,
        which I think is a better comparison. Normalized by final price 
        for losses, but by initial price for gains.
        
        This metric therefore punishes losses much greater than annualized ROI.
        """
        index = self._roi_interval < 0
        new = self._roi_interval.copy()
        new[index] = self._rof_interval[index]
        
        annualized = (1 + new)**(365 / self.times_delta) - 1
        return annualized
    
    
    
        
        

from backtester.stockdata import YahooData2

y = YahooData2()
df = y.dataframes['TQQQ']

r = ROI(df['Adj Close'], 100)

r.index_interval_starts
r.annualized
# r.times_delta
