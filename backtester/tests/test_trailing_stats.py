# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from backtester.indicators import (TrailingStats, 
                                   array_windows,
                                   trailing_percentiles,
                                   _array_windows)


from backtester.definitions import DF_ADJ_CLOSE, DF_VOLUME, DF_HIGH, DF_LOW
from datasets.symbols import ALL
from backtester.stockdata import YahooData


import timeit


def test1():
    
    symbols = np.array(ALL)
    np.random.seed(1)
    np.random.shuffle(symbols)

    
    # df = get_data.read_dataframe('MSFT')
    # df = get_data.read_dataframe('DIS')
    # df = get_data.read_dataframe('VOO')
    # df = get_data.read_dataframe('NUE')
    # df = get_data.read_dataframe('ROK')
    
    y = YahooData([symbols[0]])
    df = y.dataframes[symbols[0]]
    
    df1 = df.iloc[-50:]
    series = df1[DF_ADJ_CLOSE]
    window_size = 21
    ts = TrailingStats(series, window_size)
    
    
    times = ts.series.index
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.title(symbols[0] + ' Rolling Avg')
    plt.plot(times, ts.rolling_avg)
    plt.plot(series.index, series)
    plt.plot(series.index, df1[DF_HIGH])
    plt.plot(series.index, df1[DF_LOW])
    plt.grid()

    
    def test1():
        return ts.slope_normalized
    
    def test2():
        return ts._slope_normalized_check
    
    
    t1 = timeit.timeit(test1)
    print(f'Slope norm v1 time = {t1:.2f}')
    t2 = timeit.timeit(test2)
    print(f'Slope norm v2 time = {t2:.2f}')
    
    
    plt.subplot(2,1,2)
    plt.title('Exp Growth Rate')
    plt.plot(times, ts.exp_growth, label='exp rate')
    
    plt.plot(times, ts.slope_normalized, label='slope-norm')
    plt.legend()
    plt.grid()

    # pdb.set_trace()
    # plt.plot(times, ts._slope_normalized_check, '--')
    
    
    y1 = ts.slope_normalized
    y2 = ts._slope_normalized_check
    assert(np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)])))

    y1 = ts.exponential_regression[0]
    y2 = ts._exponential_regression_check[0]
    assert(np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)])))

    y1 = ts.exponential_regression[1]
    y2 = ts._exponential_regression_check[1]
    assert(np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)])))

    
    return





    
def test_trailing_avg():
    
    symbol = 'VOO'
    y = YahooData([symbol])
    df = y.dataframes[symbol]
    
    
    series = df[DF_ADJ_CLOSE].iloc[-100:]
    window = 21
    ts = TrailingStats(series, window)

    avgs = []
    for interval in ts._adj_close_intervals:
        out = np.mean(interval)
        avgs.append(out)
    avgs = np.array(avgs)
    
    assert np.all(np.isclose(avgs, ts.rolling_avg[window:]))
    
    times = series.index[window:]
    
    plt.figure()
    plt.plot(ts.series, label='actual')
    plt.plot(times, avgs, label='TEST AVG')
    plt.plot(ts.series.index, ts.rolling_avg, '--', label='IMPLEMENTED AVG')
    plt.legend()
    
    
def test_close_intervals():
    
    
    symbol = 'VOO'
    y = YahooData([symbol])
    df = y.dataframes[symbol].iloc[-400:]
    series = df[DF_ADJ_CLOSE]

    window_size = 11
    ssc = TrailingStats(series, window_size)    
    interval = ssc._adj_close_intervals[0]
    correct = series.iloc[0 : window_size]
    assert np.all(np.isclose(interval, correct))
    
    interval = ssc._adj_close_intervals[-1]
    correct = series.iloc[-window_size-1 : -1]
    assert np.all(np.isclose(interval, correct))
    

def test_intervals():
    
    def _time_days_int_intervals(times, window_size):
        """Test WITHOUT NUMBA. Construct time intervals for windowing."""
        tnum = len(times) - window_size
        intervals = []
        for ii in range(tnum):
            interval = times[ii : ii + window_size]
            intervals.append(interval)
        return intervals
    
    
    times = np.linspace(1, 100, 10000)
    window_size = 20
    _ = array_windows(times, window_size)
    
    
    
    def test1():
        return array_windows(times, window_size)
    
    def test2():
        return _time_days_int_intervals(times, window_size)
    
    def test3():
        return _array_windows(times, window_size)
    
    time1 = timeit.timeit(test1, number=400)
    time2 = timeit.timeit(test2, number=400)
    time3 = timeit.timeit(test3, number=400)
    
    print('Numba interval speed = ', time1)
    print('Python interval speed = ', time2)
    print('Numba 2  interval speed = ', time3)
    assert time1 < time2



def test_std():
    y = YahooData()
    df = y.dataframes['VOO']
    
    
    date1 = np.datetime64('2020-01-01')
    date2 = np.datetime64('2020-11-01')

    ii = (df.index > date1) & (df.index < date2)
    df = df.loc[ii]
    
    
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, window_size=10)
    
    c = ts.exp_std_dev
    series2 = pd.Series(data=c, index=series.index)
    # p = trailing_percentiles(c, window=300)

    
    # c = ts.exp_accel
    x = series.index
    y = series.values
    ax = plt.subplot(2,1,1)
    ax.set_yscale('log')
    plt.scatter(x,y,c=c, s=4)
    plt.grid()


    plt.subplot(2,1,2)
    plt.plot(x, c)
    # plt.plot(x, p/100)
    plt.grid()
    return


def test_max_loss():
    y = YahooData()
    df = y.dataframes['VOO']
    
    
    date1 = np.datetime64('2020-01-01')
    date2 = np.datetime64('2020-11-01')

    ii = (df.index > date1) & (df.index < date2)
    df = df.loc[ii]
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, window_size=10)
    max_loss = ts.max_loss
    return

    
    

if __name__ == '__main__':
    test1()
    test_trailing_avg()
    test_close_intervals()
    test_intervals()
    
    test_std()
    test_max_loss()

