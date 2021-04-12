# -*- coding: utf-8 -*-
import math
from functools import cached_property

import pandas as pd
import numpy as np
from scipy.stats import linregress

from backtester.definitions import DF_ADJ_CLOSE


def moving_avg(period, df: pd.DataFrame):
    close = df[DF_ADJ_CLOSE]
    return close.rolling(period).mean()
    


class TrailingStats:
    def __init__(self, series : pd.Series, window_size : int, indicators=()):
        self.window_size = window_size
        self.series = series   
        self.indicators = indicators
        
                
    @cached_property
    def time_days_int(self):
        """Return time in days as array[int] from start day."""
        tdelta = self.series.index - self.series.index[0]
        tdelta = tdelta.astype('timedelta64[D]')
        return tdelta
    
    
    def _append_nan(self, arr):
        """Append nan to beginning of array for window."""
        nans = np.empty(self.window_size)
        nans[:] = np.nan
        return np.append(nans, arr)        
    
    
    @cached_property
    def _time_days_int_intervals(self):
        """Construct time intervals for windowing."""
        times = self.time_days_int
        tnum = len(times) - self.window_size
        intervals = []
        for ii in range(tnum):
            interval = times[ii : ii + self.window_size]
            intervals.append(interval)
        return intervals
        
        
    @cached_property
    def _adj_close_intervals(self) -> list[pd.Series]:
        """list[pandas.Series] : Close Intervals on where to calculate statistics."""
        tnum = len(self.series.index) - self.window_size
        series = self.series
        intervals = []
        for ii in range(tnum):
            interval = series.iloc[ii : ii + self.window_size]
            intervals.append(interval)
        return intervals     
    
    
    def _regression(self, x, y):
        x = np.asarray(x)
        xmean = np.mean(x, axis=1)
        ymean = np.mean(y, axis=1)        
        x1 = x - xmean[:, None]
        y1 = y - ymean[:, None]
        ss_xx = np.sum(x1**2, axis=1)
        ss_xy = np.sum(x1 * y1, axis=1)
        # ss_yy = np.sum(y1**2, axis=1)
        
        m = ss_xy / ss_xx
        b = ymean - m * xmean
        
        
        m = self._append_nan(m)
        b = self._append_nan(b)
        return m, b
    
    
    
    @cached_property
    def _exponential_regression_check(self):
        """Fit to y = C*exp(m*t)"""
        
        interval : pd.Series
        slopes = []
        intercepts = []
        closes = self._adj_close_intervals
        times = self._time_days_int_intervals
        
        for time, interval in zip(times, closes):
            y = np.log(interval.values)
            result = linregress(time, y)
            slopes.append(result.slope)            
            intercepts.append(math.exp(result.intercept))
            
        slopes =  self._append_nan(np.asarray(slopes))
        intercepts = self._append_nan(np.asarray(intercepts))
        return slopes, intercepts
    
    
    @cached_property
    def exponential_regression(self):
        closes = self._adj_close_intervals
        x = self._time_days_int_intervals
        y = np.log(closes)
        slopes, b = self._regression(x, y)
        intercepts = np.exp(b)
        return slopes, intercepts
        
    
    @cached_property
    def linear_regression(self):
        """(np.array, np.array) : Slope and intercept within window"""
        closes = self._adj_close_intervals
        times = self._time_days_int_intervals
        slopes, b = self._regression(times, closes)
        return slopes, b
        
        
    @cached_property
    def _slope_normalized_check(self):
        m, y0 = self.linear_regression
        y = self.rolling_avg
        return m / y
    

    @cached_property
    def slope_normalized(self):
        """Slope normalized by mean value"""
        x = np.array(self._time_days_int_intervals)
        y = self._adj_close_intervals
        
        xmean = np.mean(x, axis=1)
        ymean = np.mean(y, axis=1)
        x1 = x - xmean[:, None]
        y1 = y - ymean[:, None]
        ss_xx = np.sum(x1**2, axis=1)
        ss_xy = np.sum(x1 * y1, axis=1)
        
        m = ss_xy / ss_xx
        m = self._append_nan(m)
        yavg = self.rolling_avg
        return m / yavg
    
    
    @cached_property
    def rolling_avg(self):
        arr = np.asarray(self._adj_close_intervals)
        smoothed = np.mean(arr, axis=1)
        return self._append_nan(smoothed)
    
    
    @property
    def exp_growth(self):
        """Exponential growth rate."""
        return self.exponential_regression[0]
    
  