# -*- coding: utf-8 -*-

"""
Get a list of valid trading days.
"""

# -*- coding: utf-8 -*-
import datetime
import math
from functools import lru_cache, cached_property

from sqlalchemy import create_engine
import pandas as pd
import numpy as np

import scipy
import scipy.ndimage
import scipy.stats
from scipy.stats import linregress
from scipy.ndimage import uniform_filter

from numba import jit


from dumbot.definitions import (
    CONNECTION_PATH, 
    DF_DATE, DF_ADJ_CLOSE, DF_SMOOTH_CHANGE, DF_SMOOTH_CLOSE,
    DF_TRUE_CHANGE,
    )


engine = create_engine(CONNECTION_PATH, echo=False)


@lru_cache(maxsize=100)
def read_dataframe(symbol: str):
    """Read all available stock symbol Yahoo data."""
    dataframe = pd.read_sql(symbol, engine).set_index(DF_DATE, drop=True)
    return dataframe


    
def get_trading_days(date1 : datetime.date, date2 : datetime.date=None):
    """For two dates, get trading days in between."""
    dates = read_dataframe('DIS').index
    
    date1 = np.datetime64(date1)
    
    dates = dates[dates >= date1]
    
    if date2 is not None:
        date2 = np.datetime64(date2)
        dates = dates[dates <= date2]
    return dates


def time_days_int(time):
    """Convert dataframe time index to integer times."""
    tdelta = time - time[0]
    tdelta = tdelta.astype('timedelta64[D]')
    return tdelta


class SymbolTrailingStats:
    """Calculate trailing indicators."""
    def __init__(self, df : pd.DataFrame, window_size : int):
        self.window_size = window_size
        self.df = df        
        self.times = df.index[window_size :]
        
        
    @cached_property
    def _time_days_int(self):
        """Return time in days as array[int] from start day."""
        tdelta = self.df.index - self.df.index[0]
        tdelta = tdelta.astype('timedelta64[D]')
        return tdelta
    
    
    @cached_property
    def times_int(self):
        return self._time_days_int[self.window_size :]
    
    @cached_property
    def _time_days_int_intervals(self):
        times = self._time_days_int
        tnum = len(self.times)
        intervals = []
        for ii in range(tnum):
            interval = times[ii : ii + self.window_size]
            intervals.append(interval)
        return intervals
        
        
    @cached_property
    def _adj_close_intervals(self) -> list[pd.Series]:
        """list[pandas.Series] : Close Intervals on where to calculate statistics."""
        tnum = len(self.times)
        series = self.df[DF_ADJ_CLOSE]
        intervals = []
        for ii in range(tnum):
            interval = series.iloc[ii : ii + self.window_size]
            intervals.append(interval)
        return intervals
        
    
    @cached_property
    def linear_regression(self):
        """(np.array, np.array) : Slope and intercept within window"""
        interval : pd.Series
        slopes = []
        intercepts = []
        closes = self._adj_close_intervals
        times = self._time_days_int_intervals
        
        for time, interval in zip(times, closes):
            result = linregress(time, interval)
            slopes.append(result.slope)
            intercepts.append(result.intercept)
        return np.asarray(slopes), np.asarray(intercepts)
    
    
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
        return np.asarray(slopes), np.asarray(intercepts)
    
    
    @cached_property
    def exponential_regression(self):
        closes = self._adj_close_intervals
        x = self._time_days_int_intervals
        y = np.log(closes)
        slopes, b = self._regression(x, y)
        intercepts = np.exp(b)
        return slopes, intercepts
    
    
    @cached_property
    def momentum(self):
        """Difference of price between window_size"""
        y = self._adj_close_intervals    
        y1 = y[:, 0]
        y2 = y[:, -1]
        return y2 - y1
        
    
    @cached_property
    def rate_of_change(self):
        y = self._adj_close_intervals    
        y1 = y[:, 0]
        return  self.momentum / y1
    
    
    @cached_property
    def trailing_rolling_avg(self):
        smoothed = np.mean(self._adj_close_intervals, axis=1)
        return smoothed
    
    
    @cached_property
    def _slope_normalized_check(self):
        m, y0 = self.linear_regression
        y = self.trailing_rolling_avg
        return m / y


    @staticmethod
    def _regression(x, y):
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
        return m, b
    

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
        # ss_yy = np.sum(y1**2, axis=1)
        ss_xy = np.sum(x1 * y1, axis=1)
        

        m = ss_xy / ss_xx
        yavg = self.trailing_rolling_avg
        return m / yavg
    
    
    @property
    def exp_growth(self):
        """Exponential growth rate."""
        return self.exponential_regression[0]


    
class BestBuySell:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.time_int = time_days_int(df.index)
        self.close = df[DF_ADJ_CLOSE]
        
        
    def _calculate(self, min_hold_days=5):
        t = np.asarray(self.time_int)
        c = np.asarray(self.close)
        return get_best_buysell(t, c, min_hold_days, 2)
    
    
@jit(nopython=True)
def get_best_buysell(
        times:np.ndarray,
        closes:np.ndarray,
        min_hold_days=1,
        skip=1,):
    
    if skip > 1:
        original_len = len(times)
        times = times[::skip]
        closes = closes[::skip]
    
    length = len(times)
    newlength = int(length * (length + 1) / 2)

    buy_index = np.empty(newlength, dtype=np.int64)
    sell_index = np.empty(newlength, dtype=np.int64)
    growths = np.empty(newlength, dtype=np.float64)
    
    kk = 0
    for ii in range(length):
        buy = closes[ii]
        for jj in range(ii + min_hold_days, length):
            sell = closes[jj]
            time_diff = times[jj] - times[ii]
            growth = (math.log(sell) - math.log(buy)) / time_diff
            
            buy_index[kk] = ii
            sell_index[kk] = jj
            growths[kk] = growth
            # out[kk, :] = (ii, jj, growth)
            kk += 1
            
            
    # out = out[0 : kk]
    buy_index = buy_index[0 : kk]
    sell_index = sell_index[0 : kk]
    growths = growths[0 : kk]
    
    isort = np.argsort(growths)[::-1]
    buy_index = buy_index[isort]
    sell_index = sell_index[isort]
    growths = growths[isort]
    
    if skip > 1: 
        buy_index = buy_index * skip
        sell_index = sell_index * skip
        # ii2 = np.arange(original_len, dtype=np.int32)
        # buy_index = ii2[buy_index]
        # sell_index = ii2[sell_index]

    return buy_index, sell_index, growths

    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    