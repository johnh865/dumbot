# -*- coding: utf-8 -*-
from functools import cached_property

import pandas as pd
import numpy as np


class ROI:
    """Calculate Return on investment for multiple intervals.
    
    Parameters
    ----------
    series : pd.Series
        Closing prices with index set to time.
    interval : int
        Time interval for ROI calculation.

    """
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
        
        out = np.searchsorted(interval_counts, np.arange(imax))
        out = np.unique(out)
        return out
    
    
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
    def _roi_adjusted(self):
        """Special adjusted rate of return which I think is a better comparison.
        - Normalize by final price for losses.
        - Normalize by initial price for gains. """
        index = self._roi_interval < 0
        new = self._roi_interval.copy()
        new[index] = self._rof_interval[index]
        return new
    
    
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
        r = self._roi_adjusted
        annualized = (1 + r)**(365 / self.times_delta) - 1
        return annualized
    
    
    @cached_property
    def daily_adjusted(self):
        """ndarray[float] : Special daily adjusted rate of return."""
        r = self._roi_interval
        return (1 + r)**(1 / self.times_delta) - 1


def annualize(roi, duration: float):
    """Annualize a ROI (return on investment) given a time duration in days."""
    return (1 + roi) ** (365. / duration) - 1

    
class ROIDaily(ROI):
    """Calculate Daily Return on investment for multiple intervals.
    
    Parameters
    ----------
    series : pd.Series
        Closing prices with index set to time.    
    """
    def __init__(self, series: pd.Series, ):

        self.series = series
        self.values = self.series.values
        self.interval = 1        
        
        
    @cached_property
    def index_interval_starts(self):
        len1 = len(self.series)
        return np.arange(len1)
    


# r.times_delta
