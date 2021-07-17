# -*- coding: utf-8 -*-
import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtester.stockdata import YahooData
from backtester.indicators import TrailingIntervals, MonthlyStats
from backtester.definitions import DF_ADJ_CLOSE

def test_1():
    
    
    date1 = np.datetime64('2020-01-01')
    date2 = np.datetime64('2020-11-01')
    dates = np.arange(date1, date2)
    window_size = 10
    
    x = np.arange(len(dates))
    series = pd.Series(x, index=dates)
    
    ti = TrailingIntervals(series, window_size=window_size)
    start = ti._index_start
    end = ti._index_end
    intervals = ti._values_intervals
    
    
    x1 = intervals.ravel()
    assert intervals.shape[1] == window_size + 1
    assert np.all(intervals[0] == np.arange(window_size + 1))
    assert np.all(intervals[1] == np.arange(window_size, 2 * window_size + 1))



def test_monthly():
    """Create a function whose slope is equal to the month of the year.
    Use this to test and make sure MonthlyStats is working correctly
    for the right month. 
    """
    date1 = np.datetime64('2019-01-01')
    date2 = np.datetime64('2020-01-02')
    dates = np.arange(date1, date2)
    dates2 = pd.DatetimeIndex(dates)
    # month_days = dates2.day    

    times = (dates - dates[0]).astype('timedelta64[D]').astype(float)
    
    y = 0
    new = []
    for time, date in zip(times, dates2):
        day = date.day
        month = date.month
        new.append(y)
        y = y + month
        
    new = np.array(new)
    # plt.plot(dates, new, '.-')
    # plt.grid()
    
    series = pd.Series(new, index=dates2)
    ms = MonthlyStats(series)
    m, b, r = ms.linear_regression
    
    assert np.all(m == np.arange(1, 13))
    assert np.all(r == 1)
    
    
    
if __name__ == '__main__':
    test_monthly()