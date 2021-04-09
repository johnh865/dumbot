# -*- coding: utf-8 -*-
import pdb
import numpy as np
import matplotlib.pyplot as plt

from dumbot.indicators import TrailingStats
from dumbot.definitions import DF_ADJ_CLOSE, DF_VOLUME
from dumbot.data.symbols import ALL
from dumbot.stockdata import YahooData
import timeit


def test1():
    
    symbols = np.array(ALL)
    # np.random.seed(0)
    np.random.shuffle(symbols)

    
    # df = get_data.read_dataframe('MSFT')
    # df = get_data.read_dataframe('DIS')
    # df = get_data.read_dataframe('VOO')
    # df = get_data.read_dataframe('NUE')
    # df = get_data.read_dataframe('ROK')
    
    y = YahooData([symbols[0]])
    df = y.get_symbol_all(symbols[0])
    
    
    series = df[DF_ADJ_CLOSE]
    window_size = 301
    ts = TrailingStats(series, window_size)
    
    
    times = ts.series.index
    plt.subplot(3,1,1)
    plt.title(symbols[0] + ' Rolling Avg')
    plt.plot(times, ts.rolling_avg)
    plt.plot(series.index, series)
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.title('Exp Growth Rate')
    plt.plot(times, ts.exp_growth)
    plt.grid()
    
    plt.subplot(3,1,3)
    plt.title('Relative Slope')
    
    def test1():
        return ts.slope_normalized
    
    def test2():
        return ts._slope_normalized_check
    
    
    t1 = timeit.timeit(test1)
    print(f'Slope norm v1 time = {t1:.2f}')
    t2 = timeit.timeit(test2)
    print(f'Slope norm v2 time = {t2:.2f}')

    plt.plot(times, ts.slope_normalized)
    # pdb.set_trace()
    plt.plot(times, ts._slope_normalized_check, '--')
    
    
    y1 = ts.slope_normalized
    y2 = ts._slope_normalized_check
    assert(np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)])))

    y1 = ts.exponential_regression[0]
    y2 = ts._exponential_regression_check[0]
    assert(np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)])))

    y1 = ts.exponential_regression[1]
    y2 = ts._exponential_regression_check[1]
    assert(np.all(np.isclose(y1[~np.isnan(y1)], y2[~np.isnan(y2)])))

    plt.grid()
    
    return





    
def test_trailing_avg():
    
    symbol = 'VOO'
    y = YahooData([symbol])
    df = y.get_symbol_all(symbol)    
    series = df[DF_ADJ_CLOSE]
    window = 101
    ts = TrailingStats(series, window)

    avgs = []
    for interval in ts._adj_close_intervals:
        out = np.mean(interval)
        avgs.append(out)
    avgs = np.array(avgs)
    
    assert np.all(np.isclose(avgs, ts.rolling_avg[window:]))
    
    times = series.index[window:]
    plt.plot(ts.series, label='actual')
    plt.plot(times, avgs, label='TEST AVG')
    plt.plot(ts.series.index, ts.rolling_avg, '--', label='IMPLEMENTED AVG')
    plt.legend()
    
    
def test_close_intervals():
    
    
    symbol = 'VOO'
    y = YahooData([symbol])
    df = y.get_symbol_all(symbol)    
    series = df[DF_ADJ_CLOSE]

    window_size = 11
    ssc = TrailingStats(series, window_size)    
    interval = ssc._adj_close_intervals[0]
    correct = series.iloc[0 : window_size]
    assert np.all(np.isclose(interval, correct))
    
    interval = ssc._adj_close_intervals[-1]
    correct = series.iloc[-window_size-1 : -1]
    assert np.all(np.isclose(interval, correct))
    




def test_buysell():
    
    
    symbol = 'BBY'
    y = YahooData([symbol])
    df = y.get_symbol_all(symbol)    
    series = df[DF_ADJ_CLOSE]

    
    df = df.iloc[-1000:]
    bbs = get_data.BestBuySell(df)
    buy, sell, growth = bbs._calculate(20)

        
    ii = 0
    plt.subplot(2,1,1)
    for buy1, sell1 in zip(buy, sell):
        times = df.index[[buy1, sell1]]
        closes = bbs.close[[buy1, sell1]]
        plt.plot(times, closes, '--', alpha=.5)
        ii += 1
        if ii > 5000:
            break
        
    plt.plot(bbs.df.index, bbs.close, alpha=1)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(bbs.df.index, df[DF_VOLUME], alpha=1)
    plt.grid()
    # pdb.set_trace()
    return


if __name__ == '__main__':
    # test1()
    test_trailing_avg()
    test_close_intervals()

