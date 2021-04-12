# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData
from backtester import utils
from backtester.analysis import avg_future_growth
 

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

def test_future():
    
    symbol = 'VOO'
    y = YahooData([symbol])
    df = y.get_symbol_all(symbol)    
    df = df.iloc[-1000::2]
    
    series = df[DF_ADJ_CLOSE]
    dates = series.index
    times = utils.dates2days(series.index)
    prices = series.values
    
    
    _, growth = avg_future_growth(times, prices, window=20)


    plt.plot(dates, prices, alpha=.2)
    
    plt.scatter(dates[0 : -20],
                prices[0 : -20], 
                c=growth*100, s=20, cmap='coolwarm', 
                vmin=-1., vmax=1.)
    plt.colorbar()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Price')
    return


if __name__ == '__main__':
    test_future()