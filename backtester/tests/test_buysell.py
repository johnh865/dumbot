# -*- coding: utf-8 -*-
import pdb
import numpy as np
import matplotlib.pyplot as plt

from backtester.definitions import DF_ADJ_CLOSE, DF_VOLUME
from backtester.stockdata import YahooData
from backtester.analysis import buysell, BuySell
from backtester import utils



def test_buysell():
    
    symbol = 'VOO'
    y = YahooData([symbol])
    df = y.get_symbol_all(symbol)    
    df = df.iloc[-1000::2]
    
    series = df[DF_ADJ_CLOSE]
    
    times = utils.dates2days(series.index)
    prices = series.values
    
    buy, sell, growth, change = buysell(times, prices, 
                                             min_hold_days=5,
                                             max_hold_days=100)
    
    num = len(buy)
    top = int(num/25)

        
    ii = 0
    plt.subplot(2,1,1)
    for buy1, sell1 in zip(buy, sell):
        times = series.index[[buy1, sell1]]
        closes = series[[buy1, sell1]]
        plt.plot(times, closes, '--', alpha=.5)
        ii += 1
        if ii > top:
            break
        
    plt.plot(series.index, series.values, alpha=1)
    plt.grid()

    # pdb.set_trace()
    return

def test_2():
    
    symbol = 'VOO'
    y = YahooData([symbol])
    df = y.get_symbol_all(symbol)    
    df = df.iloc[-1000::2]
    series = df[DF_ADJ_CLOSE]
    times = utils.dates2days(series.index)
    prices = series.values
    
    bs = BuySell(times, prices, min_hold_days=5, max_hold_days=50)
    bs.data
    bs.max_change
    
    
if __name__ == '__main__':
    # test_buysell()
    test_2()