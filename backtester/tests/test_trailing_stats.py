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
    plt.plot(ts.times, ts.rolling_avg)
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
    plt.plot(ts.times, ts.exp_growth, label='exp rate')
    
    plt.plot(ts.times, ts.slope_normalized, label='slope-norm')
    plt.legend()
    plt.grid()

    # pdb.set_trace()
    # plt.plot(times, ts._slope_normalized_check, '--')
    
    
    y1 = ts.slope_normalized
    y2 = ts._slope_normalized_check
    assert np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)]))

    y1 = ts.exponential_regression[0]
    y2 = ts._exponential_regression_check[0]
    assert np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)]))

    y1 = ts.exponential_regression[1]
    y2 = ts._exponential_regression_check[1]
    assert np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)]))

    
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
    
    assert np.all(np.isclose(avgs, ts.rolling_avg))
    
    times = series.index[window:]
    
    plt.figure()
    plt.plot(ts.series, label='actual')
    plt.plot(ts.times, avgs, label='TEST AVG')
    plt.plot(ts.times, ts.rolling_avg, '--', label='IMPLEMENTED AVG')
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
    correct = series.iloc[-window_size :]    
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
    series2 = pd.Series(data=c, index=ts.times)
    # p = trailing_percentiles(c, window=300)

    
    # c = ts.exp_accel
    x = ts.times
    y = ts.values
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


def test_skip0():
    y = YahooData()
    df = y.dataframes['SPY']
    series = df[DF_ADJ_CLOSE]
    skip = 10
    windows_size = 50
    ts = TrailingStats(series, window_size=windows_size)
    times1 = ts.times[:: skip]
    slopes1 = ts.slope_normalized[:: skip]
    slopes1[np.isnan(slopes1)] = 0

    ts2 = TrailingStats(series, window_size=windows_size, skip=skip)
    times2 = ts2.times
    slopes2 = ts2.slope_normalized
    slopes2[np.isnan(slopes2)] = 0
    
    assert np.all(times1 == times2)
    assert np.all(slopes1 == slopes2)



def test_skip1():
    

    
    
    y = YahooData()
    df = y.dataframes['SPY']
    series = df[DF_ADJ_CLOSE]
    skip = 10
    window_size = 5
    
    def compare_attr(stats1: TrailingStats,
                     stats2: TrailingStats,
                     name: str):
        print(f'comparing {name}')
        
        
        a1 = getattr(stats1, name)[:: skip]
        a2 = getattr(stats2, name)
        a1[np.isnan(a1)] = 0
        a2[np.isnan(a2)] = 0
        
        l1 = len(a1)
        l2 = len(a2)
        print(f'len1 = {l1}')
        print(f'len2 = {l2}')
        assert np.all(a1 == a2)
            
    ts1 = TrailingStats(series, window_size=window_size)
    
    for skip in range(2, 10):
        print(f'testing skip {skip}')
        
        ts2 = TrailingStats(series, window_size=window_size, skip=skip)

        
        
        # # Check rolling avg.
        # rolling1 = ts1._adj_close_intervals.mean(axis=1)
        
        # rolling1 = ts1._append_nan(rolling1)[:: skip]
        # rolling1[np.isnan(rolling1)] = 0 
        # rolling11 = ts1.rolling_avg[:: skip]
        # rolling11[np.isnan(rolling11)] = 0
        
        # assert np.all(rolling1 == rolling11)
        
        rolling2 = ts2._adj_close_intervals.mean(axis=1)
        rolling2[np.isnan(rolling2)] = 0 
            
        compare_attr(ts1, ts2, 'times')
        compare_attr(ts1, ts2, 'time_days_int')
        compare_attr(ts1, ts2, 'rolling_avg')
        compare_attr(ts1, ts2, 'return_ratio')
        compare_attr(ts1, ts2, 'slope_normalized')
        


def test_skip():
    y = YahooData()
    df = y.dataframes['SPY']
    series = df[DF_ADJ_CLOSE]
    
    for skip in range(2, 43, 3):
        
        ts = TrailingStats(series, window_size=20, skip=skip)
        slope = ts.slope_normalized
        # print(len(slope))
        # print(len(ts.times))
        assert len(slope) == len(ts.times)

    

if __name__ == '__main__':
    test1()
    test_trailing_avg()
    test_close_intervals()
    test_intervals()
    
    test_std()
    test_max_loss()
    test_skip1()
    test_skip()
    # # test_skip0()

