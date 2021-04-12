# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import backtester
from backtester.stockdata import YahooData
from backtester.definitions import (
    DF_ADJ_CLOSE, DF_CLOSE, DF_OPEN, DF_HIGH, DF_LOW, DF_VOLUME)

symbol = 'APOL'
yd = YahooData()
df = yd.get_symbol_all(symbol)


def plot(df: pd.DataFrame):
    s_close = df[DF_CLOSE]
    s_open = df[DF_OPEN]
    s_high = df[DF_HIGH]
    s_low = df[DF_LOW]
    s_time = df.index
    
    ax = plt.gca()    
    plt.xticks(rotation=60, ha='right')
    plt.grid(color='k', lw=0.2, alpha=0.2)
    
    months = matplotlib.dates.MonthLocator()
    
    ax.xaxis.set_major_locator(months)
    year_month = matplotlib.dates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(year_month)
    plt.plot(s_time, s_close)
    
    

if __name__ == '__main__':
    y = YahooData()
    df = y.get_symbol_all('MSFT')
    plot(df)