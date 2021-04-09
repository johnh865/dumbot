# -*- coding: utf-8 -*-
import pdb
import numpy as np
from dumbot.stockdata import get_data
from dumbot.definitions import DF_ADJ_CLOSE, DF_VOLUME
import matplotlib.pyplot as plt
from dumbot.stockdata.symbols import ALL
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
    df = get_data.read_dataframe(symbols[0])
    
    
    ssc = get_data.SymbolTrailingStats(df, 301)
    
    ssc._time_days_int
    times = ssc.times
    plt.subplot(3,1,1)
    plt.title(symbols[0] + ' Rolling Avg')
    plt.plot(times, ssc.trailing_rolling_avg)
    plt.plot(ssc.df.index, ssc.df[DF_ADJ_CLOSE])
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.title('Exp Growth Rate')
    plt.plot(times, ssc.exp_growth)
    plt.grid()
    
    plt.subplot(3,1,3)
    plt.title('Relative Slope')
    
    def test1():
        return ssc.slope_normalized
    
    def test2():
        return ssc._slope_normalized_check
    
    
    t1 = timeit.timeit(test1)
    print(f'Slope norm v1 time = {t1:.2f}')
    t2 = timeit.timeit(test2)
    print(f'Slope norm v2 time = {t2:.2f}')

    plt.plot(times, ssc.slope_normalized)
    # pdb.set_trace()
    plt.plot(times, ssc._slope_normalized_check, '--')
    
    assert(np.all(np.isclose(
        ssc.slope_normalized, 
        ssc._slope_normalized_check
        )))
    
    assert(np.all(np.isclose(
        ssc.exponential_regression[0], 
        ssc._exponential_regression_check[0]
        )))    
    
    assert(np.all(np.isclose(
        ssc.exponential_regression[1], 
        ssc._exponential_regression_check[1]
        )))        
    
    plt.grid()
    
    return

    
    
    
def test_trailing_avg():
    
    df = get_data.read_dataframe('VOO')
    ssc = get_data.SymbolTrailingStats(df, 101)

    avgs = []
    for interval in ssc._adj_close_intervals:
        out = np.mean(interval)
        avgs.append(out)
    avgs = np.array(avgs)
    
    assert np.all(np.isclose(avgs, ssc.trailing_rolling_avg))
    
    
    plt.plot(ssc.df[DF_ADJ_CLOSE], label='actual')
    plt.plot(ssc.times, avgs, label='TEST AVG')
    plt.plot(ssc.times, ssc.trailing_rolling_avg, label='IMPLEMENTED AVG')
    plt.legend()
    
    
def test_close_intervals():
    df = get_data.read_dataframe('VOO')
    window_size = 11
    ssc = get_data.SymbolTrailingStats(df, window_size)    
    interval = ssc._adj_close_intervals[0]
    correct = df[DF_ADJ_CLOSE].iloc[0 : window_size]
    assert np.all(np.isclose(interval, correct))
    
    interval = ssc._adj_close_intervals[-1]
    correct = df[DF_ADJ_CLOSE].iloc[-window_size-1 : -1]
    assert np.all(np.isclose(interval, correct))
    
    assert df.index[window_size] == ssc.times[0]



def test_buysell():
    df = get_data.read_dataframe('BBY')
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
    test_buysell()
    
    
    
    
    
    
    
    
    
    