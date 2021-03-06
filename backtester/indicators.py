# -*- coding: utf-8 -*-
import math
from functools import cached_property, wraps
from numba import njit
from math import sqrt, ceil

import pandas as pd
import numpy as np
from scipy.stats import linregress
from backtester.exceptions import NotEnoughDataError
from backtester.utils import round_to_quarters
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
    tnum = xlen - window + 1
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
    tnum = ceil((xlen - window + 1) / skip)
    tnum = int(tnum)
    
    if tnum < 1:
        raise NotEnoughDataError('Window size is greater than length x')
   
        
    intervals = np.zeros((tnum, window))
    for ii in range(tnum):
        kk = ii * skip
        intervals[ii] = x[kk : kk + window]
    return intervals    





# @njit
def array_windows_index(x: np.ndarray,
                        window: np.int64, 
                        index_start: np.ndarray, 
                        index_end: np.ndarray):
    ilen1 = len(index_start)
    ilen2 = len(index_end)
    # print(len(x), window)
    intervals = np.zeros((ilen1, window))
    intervals[:] = np.nan
    for ii, (k_start, k_end) in enumerate(zip(index_start, index_end)):    
        intervals[ii, 0 : window] = x[k_start : k_end]
    
    return intervals


def array_intervals_index(x: np.ndarray, 
                          index_start: np.ndarray,
                          index_end: np.ndarray):
    ilen1 = len(index_start)
    ilen2 = len(index_end)
    col_lengths = np.max(index_end - index_start)
    intervals = np.zeros((ilen1, col_lengths))
    intervals[:] = np.nan
    for ii, (k_start, k_end) in enumerate(zip(index_start, index_end)):    
        intervals[ii, 0 : (k_end - k_start)] = x[k_start : k_end]
    return intervals


def append_nan(arr, window: int):
    nans = np.empty(window - 1)
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
    def __init__(self, series : pd.Series, 
                 window_size : int, skip: int=1):
        self.window_size = np.int64(window_size)
        self.skip = np.int64(skip)
        self.series = series   
        # self._start_loc = self.window_size - 1
    
        
    @cached_property
    def _index_end(self) -> np.ndarray:
        start = self.window_size
        end = len(self.series) + 1
        return np.arange(start, end, self.skip)
    
    
    @cached_property
    def _index_start(self) -> np.ndarray:
        return self._index_end - self.window_size
    
    
    @cached_property
    def _index_display(self) -> np.ndarray:
        """Index which marks time points to be output as representative of inteval."""
        return self._index_end - 1
        
    
        
    @cached_property
    def times(self) -> np.ndarray:
        """Associated times for output properties."""
        # return self.series.index.values[self._start_loc :: self.skip]
        return self.series.index.values[self._index_display]
    
    
    @cached_property
    def values(self) -> np.ndarray:
        """Associated values with times."""
        # return self.series.values[self._start_loc :: self.skip]    
        return self.series.values[self._index_display]    
    
    
    @cached_property
    def time_days_int(self):
        """Return time in days as array[int] from start day."""
        # return self._time_days_int_noskip[self._start_loc :: self.skip]    
        return self._time_days_int_noskip[self._index_display]    
    
    
    @cached_property
    def _time_days_int_noskip(self):
        tdelta = self.series.index - self.series.index[0]
        try:
            tdelta = tdelta.astype('timedelta64[D]')
        except TypeError:
            pass
        return np.asarray(tdelta)
        
    
    @cached_property
    def _time_days_int_intervals(self):
        """Construct time intervals for windowing."""
        times = self._time_days_int_noskip
        return array_windows_index(times, 
                                   self.window_size, 
                                   self._index_start,
                                   self._index_end,)
        
        # if self.skip > 1:
        #     return array_windows_skip(times, self.window_size, self.skip)
        # else:
        #     return array_windows(times, self.window_size)

        
    @cached_property
    def _values_intervals(self) -> np.ndarray:
        """list[pandas.Series] : Close Intervals on where to calculate statistics."""
        # tnum = len(self.series.index) - self.window_size
        # series = self.series
        values = self.series.values
        return array_windows_index(values, 
                                   self.window_size, 
                                   self._index_start,
                                   self._index_end,)

        # if self.skip > 1:
        #     return array_windows_skip(values, self.window_size, self.skip)
        # else:
        #     return array_windows(values, self.window_size)
    



        
class __TrailingBaseOLD:
    def __init__(self, series : pd.Series, 
                 window_size : int, skip: int=1):
        self.window_size = np.int64(window_size)
        self.skip = np.int64(skip)
        self.series = series   
        
        
    @cached_property
    def times(self) -> np.ndarray:
        """Associated times for output properties."""
        return self.series.index.values[:: self.skip]
    
    
    @cached_property
    def _time_days_int_noskip(self):
        tdelta = self.series.index - self.series.index[0]
        try:
            tdelta = tdelta.astype('timedelta64[D]')
        except TypeError:
            pass
        return np.asarray(tdelta)


    @cached_property
    def time_days_int(self):
        """Return time in days as array[int] from start day."""
        return self._time_days_int_noskip[:: self.skip]
        
    
    
    def _append_nan(self, arr):
        """Append nan to beginning of array for window."""
        nans = np.empty(self.window_size - 1)
        nans[:] = np.nan
        nans = nans[:: self.skip]
        return np.append(nans, arr)        
    
    
    @staticmethod
    def _append_nan_dec(func):
        """Decorator to append np.nan to beginning of the array."""
        
        @wraps(func)
        def func2(obj):
            output = func(obj)
            return obj._append_nan(output)
        return func2
    
    
    @cached_property
    def _time_days_int_intervals(self):
        """Construct time intervals for windowing."""
        times = self._time_days_int_noskip
        
        if self.skip > 1:
            return array_windows_skip(times, self.window_size, self.skip)
        else:
            return array_windows(times, self.window_size)

        
    @cached_property
    def _values_intervals(self) -> np.ndarray:
        """list[pandas.Series] : Close Intervals on where to calculate statistics."""
        # tnum = len(self.series.index) - self.window_size
        # series = self.series
        values = self.series.values
        
        if self.skip > 1:
            return array_windows_skip(values, self.window_size, self.skip)
        else:
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
        xmean = np.nanmean(x, axis=1)
        ymean = np.nanmean(y, axis=1)        
        x1 = x - xmean[:, None]
        y1 = y - ymean[:, None]
        ss_xx = np.nansum(x1**2, axis=1)
        ss_xy = np.nansum(x1 * y1, axis=1)
        ss_yy = np.nansum(y1**2, axis=1)
        
        m = ss_xy / ss_xx
        b = ymean - m * xmean
        r = ss_xy / np.sqrt(ss_xx * ss_yy)

        # m = self._append_nan(m)
        # b = self._append_nan(b)
        # r = self._append_nan(r)
        return m, b, r
    

    
    @cached_property
    def _exponential_regression_check(self):
        """Fit to y = C*exp(m*t)"""
        
        interval : pd.Series
        slopes = []
        intercepts = []
        closes = self._values_intervals
        times = self._time_days_int_intervals
        
        for time, interval in zip(times, closes):
            y = np.log(interval)
            result = linregress(time, y)
            slopes.append(result.slope)            
            intercepts.append(math.exp(result.intercept))
            
        # slopes =  self._append_nan(np.asarray(slopes))
        # intercepts = self._append_nan(np.asarray(intercepts))
        slopes = np.array(slopes)
        intercepts = np.array(intercepts)
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
        closes = self._values_intervals
        x = self._time_days_int_intervals
        y = np.log(closes)
        slopes, b, _ = self._regression(x, y)
        intercepts = np.exp(b)
        return slopes, intercepts
        
    
    @cached_property
    def linear_regression(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """(np.array, np.array) : Slope, intercept, and r-value within window"""
        closes = self._values_intervals
        times = self._time_days_int_intervals
        slopes, b, r = self._regression(times, closes)
        return slopes, b, r
        
        
    @cached_property
    def _slope_normalized_check(self):
        m, y0, r = self.linear_regression
        y = self.mean
        return m / y
    

    @cached_property
    def slope_normalized(self):
        """Slope normalized by mean value"""
        x = np.array(self._time_days_int_intervals)
        y = self._values_intervals
        
        xmean = np.nanmean(x, axis=1)
        ymean = np.nanmean(y, axis=1)
        x1 = x - xmean[:, None]
        y1 = y - ymean[:, None]
        ss_xx = np.nansum(x1**2, axis=1)
        ss_xy = np.nansum(x1 * y1, axis=1)
        
        m = ss_xy / ss_xx
        # m = self._append_nan(m)
        yavg = self.mean
        return m / yavg
    
    
    @cached_property
    # @TrailingBase._append_nan_dec
    def mean(self):
        arr = np.asarray(self._values_intervals)
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
        y_true = self.values
        return (y_true - y_reg) / y_reg
        
    
    @cached_property
    # @TrailingBase._append_nan_dec
    def exp_std_dev(self):
        """Standard deviation of window considering exponential fit.
        Try to measure volatility."""
        
        y_reg = self.exp_reg_value[:, None]
        y_intervals = self._values_intervals
        delta = (y_intervals - y_reg) / y_reg
        return np.nanstd(delta, axis=1)
    
    
    @cached_property
    def std_dev(self):
        return np.nanstd(self._values_intervals, axis=1)
    
    
    
    
    @cached_property
    # @TrailingBase._append_nan_dec
    def max_loss(self):
        """Max loss of current day considering previous days."""
        closes = self._values_intervals
        maxes = np.nanmax(closes, axis=1)
        values = self.series.values
        last = values[self._index_end  - 1]
        return (maxes - last) / maxes
    
    
    @cached_property
    # @TrailingBase._append_nan_dec
    def max_gain(self):
        """Max loss of current day considering previous days."""
        closes = self._values_intervals
        mins = np.nanmin(closes, axis=1)
        
        values = self.series.values
        last = values[self._index_end  - 1]
        return (last - mins) / mins    
    
    
    @cached_property
    # @TrailingBase._append_nan_dec
    def return_ratio(self):
        """Get ratio of start to end value."""
        values = self.series.values
        start = values[self._index_start]
        last = values[self._index_end  - 1]
        return (last - start) / start
    
         

class FutureStats(TrailingStats):
    @cached_property
    def _index_display(self) -> np.ndarray:
        """Index which marks time points to be output as representative of inteval."""
        return self._index_start
    
    

class TrailingIntervals(TrailingStats):
    def __init__(self, series : pd.Series, 
                 window_size : int):
        self.window_size = np.int64(window_size)
        self.series = series   
        
    
    @cached_property
    def _get_index(self):
        times = self.series.index
        tnum = len(times)
        # leftover = tnum % self.window_size
        # indices = np.arange(tnum - 1 , -1-leftover, -self.window_size)
        indices = np.arange(0, tnum, self.window_size)
        index_start = indices[0 : -1]
        index_end = indices[1:] + 1
        return index_start, index_end


    @cached_property
    def _index_start(self) -> np.ndarray:
        return self._get_index[0]

        
    @cached_property
    def _index_end(self) -> np.ndarray:
        return self._get_index[1]

    @cached_property
    def _index_display(self) -> np.ndarray:
        """Index which marks time points to be output as representative of inteval."""
        return self._index_end
            

    @cached_property
    def _time_days_int_intervals(self):
        """Construct time intervals for windowing."""
        times = self._time_days_int_noskip
        return array_intervals_index(times, 
                                   self._index_start,
                                   self._index_end,)
        

        
    @cached_property
    def _values_intervals(self) -> np.ndarray:
        """list[pandas.Series] : Close Intervals on where to calculate statistics."""
        # tnum = len(self.series.index) - self.window_size
        # series = self.series
        values = self.series.values
        return array_intervals_index(values, 
                                   self._index_start,
                                   self._index_end,)
    


class QuarterStats(TrailingIntervals):
    def __init__(self, series : pd.Series):
        self.series = series   
        
        
    @cached_property
    def _get_index(self):
        times = self.series.index
        quarters = times.quarter.values
        years = times.year.values        
        
        first_year = years.min()
        last_year = years.max()
        quarter_range = [1,2,3,4]
        year_range = range(first_year, last_year + 1)
        
        index_start = []
        index_end = []
        
        for year in year_range:
            for quarter in quarter_range:
                bool_arr = (quarter == quarters) & (year == years)
                locs = np.where(bool_arr)[0]
                if len(locs) > 0:
                    loc_min = np.min(locs)
                    loc_max = np.max(locs) + 1
                    index_start.append(loc_min)
                    index_end.append(loc_max)
        
        index_start = np.array(index_start)
        index_end = np.array(index_end)
        return index_start, index_end
    
    
    @cached_property
    def _index_start(self) -> np.ndarray:
        return self._get_index[0]

        
    @cached_property
    def _index_end(self) -> np.ndarray:
        return self._get_index[1]

    
    
    @cached_property
    def _index_display(self) -> np.ndarray:
        """Index which marks time points to be output as representative of inteval."""
        return self._index_start



    @cached_property
    def times(self):
        
        t1 = self.series.index.values[self._index_display]
        return round_to_quarters(t1)
    
    
    

    
class MonthlyStats(QuarterStats):
    def __init__(self, series : pd.Series):
        self.series = series   
        
        
    # @cached_property
    # def _get_index(self):
    #     times = self.series.index
    #     months = times.month.values
    #     years = times.year.values        
        
    #     first_year = years.min()
    #     last_year = years.max()
    #     month_range = range(1, 13)
    #     year_range = range(first_year, last_year + 1)
        
    #     index_start = []
    #     index_end = []
        
    #     for year in year_range:
    #         for month in month_range:
    #             bool_arr = (month == months) & (year == years)
    #             locs = np.where(bool_arr)[0]
    #             if len(locs) > 0:
    #                 loc_min = np.min(locs)
    #                 loc_max = np.max(locs) + 1
    #                 index_start.append(loc_min)
    #                 index_end.append(loc_max)
        
    #     index_start = np.array(index_start)
    #     index_end = np.array(index_end)
    #     return index_start, index_end
    
    @cached_property
    def _get_index(self):
        times = self.series.index
        # tnum = len(times)
        month_days = times.day
        locs = month_days == 1
        indices = np.where(locs)[0]
        
        # indices = np.arange(0, tnum, self.window_size)
        index_start = indices[0 : -1]
        index_end = indices[1:] + 1
        return index_start, index_end
    
    
    

    @cached_property
    def times(self):
        t1 = self.series.index.values[self._index_display]
        return t1.astype('datetime64[M]')  
    

class WeeklyStats(MonthlyStats):
    """Get weekly stats using iso calendar."""
    def __init__(self, series: pd.Series, num_weeks=1):
        self.num_weeks = num_weeks
        self.series = series
        
        
    @cached_property
    def _get_index(self):
        # 52 full weeks in a year
        # 53 weeks in year, 53rd has 1 or 2 days.
        
        times = self.series.index
        calendar = times.isocalendar()

        break_indices = np.where(calendar.day == 1)[0]
        if self.num_weeks > 1:
            break_indices = break_indices[:: self.num_weeks]
        
        index_start = break_indices[0 : -1]
        index_end = break_indices[1:]


        return index_start, index_end
        


    @cached_property
    def times(self):
        t1 = self.series.index.values[self._index_display]
        return t1.astype('datetime64[D]') 
    
    
        
# x = np.arange(100)
# y = x ** 2
# s = pd.Series(y, index=x)
# t = TrailingStats(s, 11, 3)
# t = TrailingStats(s, 10)
# t.rolling_avg



# def trailing_sharpe(series: pd.Series, period, num_periods):
#     ts = TrailingStats(series, window_size=period)
    