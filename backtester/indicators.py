# -*- coding: utf-8 -*-
import math
from functools import cached_property, wraps
from numba import njit
from math import sqrt

import pandas as pd
import numpy as np
from scipy.stats import linregress
from backtester.exceptions import NotEnoughDataError

import pdb



def array_windows(x: np.ndarray, 
                  window: np.int64) -> np.ndarray:
    """Construct 2-d array for with a window interval for each x.
    
    Parameters
    ----------
    x : np.ndarray shape (a,)
        Array to construct windows.
    window : np.int64
        Size of window.

    Returns
    -------
    intervals : np.ndarray shape (a, b)
        Intervals of x.

    Raises
    ------
    backtester.exceptions.NotEnoughDataError
        Raise if window is larger than len(x).
    """
    
    # This is to catch a weird CPUDispatcher NULL when using numba in cProfile
    try:
            return _array_windows(x, window)        
    except SystemError:
        raise NotEnoughDataError('Window size is greater than length x')
        
        
def array_windows_skip(x: np.ndarray, window:np.int64, skip=1):
    try:
        return _array_windows_skip(x, window, skip)
    except SystemError:
        raise NotEnoughDataError('Window size is greater than length x')


@njit
def _array_windows(x: np.ndarray, window: np.int64) -> np.ndarray:
    """Construct 2-d array for with a window interval for each x.
    
    Parameters
    ----------
    x : np.ndarray shape (a,)
        Array to construct windows.
    window : np.int64
        Size of window.

    Returns
    -------
    intervals : np.ndarray shape (a, b)
        Intervals of x.

    Raises
    ------
    backtester.exceptions.NotEnoughDataError
        Raise if window is larger than len(x).
    """
    xlen = len(x)
    tnum = xlen - window
    if tnum < 1:
        raise NotEnoughDataError('Window size is greater than length x')
    
    intervals = np.zeros((tnum, window))
    for ii in range(tnum):
        intervals[ii] = x[ii : ii + window]
    return intervals


@njit
def _array_windows_skip(x: np.ndarray, 
                        window: np.int64,
                        skip: np.int64=1) -> np.ndarray:
    xlen = len(x)
    tnum = (xlen - window) / skip
    if tnum < 1:
        raise NotEnoughDataError('Window size is greater than length x')
        
    intervals = np.zeros((tnum, window))
    for ii in range(tnum):
        kk = ii * skip
        intervals[ii] = x[kk : kk + window]
    return intervals    


def append_nan(arr, window: int):
    nans = np.empty(window)
    nans[:] = np.nan
    return np.append(nans, arr)        



def ignore_nan(func):
    """Decorator to ignore NAN from results during calculation but add them
    back in later."""
    def func2(x):
        
        is_nan = np.isnan(x)
        not_nan = ~is_nan
        
        x2 = x[not_nan]
        out = func(x2)
        
        new = np.zeros(x.shape)
        new[is_nan] = np.nan
        new[not_nan] = out
        return new
    
    return func2

def trailing_percentiles(x: np.ndarray, window: np.int64):
    """Get percentile distributions from [ 0 to 100] for each value in x."""
    intervals = array_windows(x, window)
    tnum = len(intervals)
    out = np.zeros(tnum)
    pbins = np.arange(100)    
    
    for ii, interval in enumerate(intervals):
        percentiles = np.nanpercentile(interval, pbins)
        out[ii] = np.interp(interval[-1], percentiles, pbins)
    return append_nan(out, window)


@ignore_nan
@njit
def trailing_mean(x: np.array,):
    """Calculate mean for all data to the current point."""
    ilen, jlen = x.shape

    new = np.zeros(ilen, jlen)
    new[0, :] = np.nan
    
    for ii in range(1, ilen):
        xi = np.nanmean(x[0 : ii], axis=0)
        new[ii] = xi
    return new


@ignore_nan
def cumulative_mean(x: np.array):
    return _cumulative_mean(x)


@njit
def _cumulative_mean(x: np.array):
    """Doesn't go through ignore_nan decorator"""
    xlen = len(x)
    out = np.zeros(xlen)    
    net = 0 
    for ii in range(0, xlen):
        net += x[ii]
        out[ii] = net / (ii + 1)
    return out
    
 
    
@ignore_nan
@njit
def cumulative_std(x: np.ndarray, mean: np.ndarray=None):
    """Corrected sample std dev for all data to the current point cumulatively.
    You must also calculate the cumulative mean and use as input."""
                
    xlen = len(x)
    out = np.zeros(xlen)
    net = 0
    if mean is None:
        mean = _cumulative_mean(x)
    
    for ii in range(xlen):
        net += (x[ii] - mean[ii]) **2
        out[ii] = sqrt(net / ii)
    return out


class TrailingBase:
    def __init__(self, series : pd.Series, window_size : int):
        self.window_size = np.int64(window_size)
        self.series = series   
        
        
    @cached_property
    def times(self) -> np.ndarray:
        """Associated times for output properties."""
        return self.series.index.values[0 : -self.window_size]
        

    @cached_property
    def time_days_int(self):
        """Return time in days as array[int] from start day."""
        tdelta = self.series.index - self.series.index[0]
        try:
            tdelta = tdelta.astype('timedelta64[D]')
        except TypeError:
            pass
        return np.array(tdelta)
    
    
    def _append_nan(self, arr):
        """Append nan to beginning of array for window."""
        nans = np.empty(self.window_size)
        nans[:] = np.nan
        return np.append(nans, arr)        
    
    
    @staticmethod
    def _append_nan_dec(func):
        """Decorator to append np.nan to beginning of the array."""
        
        @wraps(func)
        def func2(self):
            output = func(self)
            return self._append_nan(output)
        return func2
    
    
    @cached_property
    def _time_days_int_intervals(self):
        """Construct time intervals for windowing."""
        times = self.time_days_int
        return array_windows(times, self.window_size)

        
    @cached_property
    def _adj_close_intervals(self) -> list[pd.Series]:
        """list[pandas.Series] : Close Intervals on where to calculate statistics."""
        # tnum = len(self.series.index) - self.window_size
        # series = self.series
        values = self.series.values
        return array_windows(values, self.window_size)
    

class TrailingStats(TrailingBase):
    """Calculate trailing statistics for a window of time. 

    Parameters
    ----------
    series : pd.Series
        times and values.
    window_size : int
        Size of window.
    """

    def _regression(self, x, y):
        """Perform regression.
        https://mathworld.wolfram.com/LeastSquaresFitting.html"""
        
        x = np.asarray(x)
        xmean = np.mean(x, axis=1)
        ymean = np.mean(y, axis=1)        
        x1 = x - xmean[:, None]
        y1 = y - ymean[:, None]
        ss_xx = np.sum(x1**2, axis=1)
        ss_xy = np.sum(x1 * y1, axis=1)
        ss_yy = np.sum(y1**2, axis=1)
        
        m = ss_xy / ss_xx
        b = ymean - m * xmean
        r = ss_xy / np.sqrt(ss_xx * ss_yy)

        m = self._append_nan(m)
        b = self._append_nan(b)
        r = self._append_nan(r)
        return m, b, r
    
    
    
    
    
    @cached_property
    def _exponential_regression_check(self):
        """Fit to y = C*exp(m*t)"""
        
        interval : pd.Series
        slopes = []
        intercepts = []
        closes = self._adj_close_intervals
        times = self._time_days_int_intervals
        
        for time, interval in zip(times, closes):
            y = np.log(interval)
            result = linregress(time, y)
            slopes.append(result.slope)            
            intercepts.append(math.exp(result.intercept))
            
        slopes =  self._append_nan(np.asarray(slopes))
        intercepts = self._append_nan(np.asarray(intercepts))
        return slopes, intercepts
    
    
    @cached_property
    def exponential_regression(self):
        """Fit to y = C*exp(m*t).
        
        Returns
        -------
        rate :
            Parameter `m` in equation y = C*exp(m*t)
        amplitude :
            Parameter `ln(C)` in equation y = C*exp(m*t)
        """
        closes = self._adj_close_intervals
        x = self._time_days_int_intervals
        y = np.log(closes)
        slopes, b, _ = self._regression(x, y)
        intercepts = np.exp(b)
        return slopes, intercepts
        
    
    @cached_property
    def linear_regression(self):
        """(np.array, np.array) : Slope and intercept within window"""
        closes = self._adj_close_intervals
        times = self._time_days_int_intervals
        slopes, b, r = self._regression(times, closes)
        return slopes, b, r
        
        
    @cached_property
    def _slope_normalized_check(self):
        m, y0, r = self.linear_regression
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
    @TrailingBase._append_nan_dec
    def rolling_avg(self):
        arr = np.asarray(self._adj_close_intervals)
        smoothed = np.mean(arr, axis=1)
        return smoothed
    
    
    @property
    def exp_growth(self):
        """Exponential growth rate."""
        return self.exponential_regression[0]
    
    @cached_property
    def exp_accel(self):
        """Exponential growth acceleration."""
        rate = self.exp_growth
        # time = self.time_days_int
        new = np.zeros(rate.shape)
        dr = rate[1:] - rate[0:-1]
        new[1:] = dr 
        new[np.isnan(new)] = 0.0
        return new
    
    
    @cached_property
    def exp_reg_value(self):
        """Fit output for exponential regression."""
        m, C = self.exponential_regression
        t = self.time_days_int.astype(float)
        out = C * np.exp(m * t)
        return out
    
    
    @cached_property
    def lin_reg_value(self):
        """Fit output for linear regression."""
        m, b = self.linear_regression
        t = self.time_days_int
        return m * t + b
    
    
    @cached_property
    def exp_reg_diff(self):
        """Difference between true value and exponential regression"""
        y_reg = self.exp_reg_value
        y_true = self.series.values
        return (y_true - y_reg) / y_reg
        
    
    @cached_property
    @TrailingBase._append_nan_dec
    def exp_std_dev(self):
        """Standard deviation of window considering exponential fit.
        Try to measure volatility."""
        
        y_reg = self.exp_reg_value[self.window_size :, None]
        y_intervals = self._adj_close_intervals
        delta = (y_intervals - y_reg) / y_reg
        return np.std(delta, axis=1)
    
    
    @cached_property
    @TrailingBase._append_nan_dec
    def max_loss(self):
        """Max loss of current day considering previous days."""
        closes = self._adj_close_intervals
        maxes = np.max(closes, axis=1)
        last = closes[:, -1]
        return (maxes - last) / maxes
    
    
    
    @cached_property
    @TrailingBase._append_nan_dec
    def max_gain(self):
        """Max loss of current day considering previous days."""
        closes = self._adj_close_intervals
        mins = np.min(closes, axis=1)
        last = closes[:, -1]
        return (last - mins) / mins    
    
    
    @cached_property
    @TrailingBase._append_nan_dec
    def return_ratio(self):
        """Get ratio of start to end value."""
        closes = self._adj_close_intervals
        start = closes[:, 0]
        last = closes[:, -1]
        return (last - start) / start
    
            
        
    


class FutureStats(TrailingStats):
    
    @cached_property
    def times(self) -> np.ndarray:
        """Associated times for output properties."""
        return self.series.index.values[self.window_size :]
    
    def _append_nan(self, arr):
        """Append nan to beginning of array for window."""
        nans = np.empty(self.window_size)
        nans[:] = np.nan
        return np.append(arr, nans)        
    
       