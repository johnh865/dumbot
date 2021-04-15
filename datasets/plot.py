# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import backtester
from backtester.stockdata import YahooData
from backtester.definitions import (
    DF_ADJ_CLOSE, DF_CLOSE, DF_OPEN, DF_HIGH, DF_LOW, DF_VOLUME)
from backtester.indicators import TrailingStats

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


symbol = 'APOL'
yd = YahooData()
df = yd.get_symbol_all(symbol)


def set_date_axes(dnum: int):
    ax = plt.gca()    
    plt.xticks(rotation=60, ha='right')
    plt.grid(which='major', color='k', lw=0.25, alpha=0.5)
    plt.grid(which='minor', color='k', lw=0.25, alpha=0.2)
    
    
    years = matplotlib.dates.YearLocator()
    months = matplotlib.dates.MonthLocator()
    days = matplotlib.dates.DayLocator()   
    
    if dnum / 12 > 20:
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_minor_locator(months)
        year_month = matplotlib.dates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(year_month)

    else:
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_minor_locator(days)
        year_month = matplotlib.dates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(year_month)
        
    
def plot(df: pd.DataFrame):
    s_close = df[DF_CLOSE]
    s_open = df[DF_OPEN]
    s_high = df[DF_HIGH]
    s_low = df[DF_LOW]
    s_time = df.index
    
    ax = plt.gca()    
    plt.xticks(rotation=60, ha='right')
    plt.grid(which='major', color='k', lw=0.25, alpha=0.5)
    plt.grid(which='minor', color='k', lw=0.25, alpha=0.2)
    
    
    years = matplotlib.dates.YearLocator()
    months = matplotlib.dates.MonthLocator()
    days = matplotlib.dates.DayLocator()
    
    dnum = len(df.index)
    if dnum / 12 > 20:
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_minor_locator(months)
        year_month = matplotlib.dates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(year_month)

    else:
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_minor_locator(days)
        year_month = matplotlib.dates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(year_month)


    ts = TrailingStats(s_close, window_size=365*2)
    ts2 = TrailingStats(s_close, window_size=100)
    s_fit = ts.lin_reg_value
    s_fit2 = ts2.lin_reg_value

    
    s3 = savgol_filter(s_close, window_length=31, polyorder=3)
    s4 = savgol_filter(s_close, window_length=1001, polyorder=3 )
    s4 = savgol_filter(s4, window_length=301, polyorder=6 )


    plt.subplot(2,1,1)
    plt.semilogy(s_time, s_close, '.-', ms=2, alpha=.2)
    plt.semilogy(s_time, s_fit, '-', alpha=.6, label='exp fit-400')
    plt.semilogy(s_time, s_fit2, '-', alpha=.6, label='exp fit-40')    
    plt.semilogy(s_time, s3, '--', alpha=.6, label='sav-gol')
    plt.semilogy(s_time, s4, '-', alpha=.6, label='sav-gol-2')
    plt.legend()
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(s_time, ts2.exp_accel)
    plt.grid()
    

    
    # plt.semilogy(s_time, ts.rolling_avg, alpha=.6, label='rolling avg')
    # pdb.set_trace()
    return




def test_smoothing(df: pd.DataFrame):
    s_close = df[DF_CLOSE]
    s_open = df[DF_OPEN]
    s_high = df[DF_HIGH]
    s_low = df[DF_LOW]
    s_time = df.index
    t2 = np.arange(len(s_time))
    windows = [4, 10, 20, 100, 200, 300, 500, 1000]
    ss = s_close
    plt.plot(t2, s_close, label='true')
    
    for window in windows:
        ss = savgol_filter(s_close, window_length = window +1, polyorder =3)
        # ss = gaussian_filter1d(ss, 10)
        plt.plot(t2, ss, '-', label=f'{window}', alpha=.5)
        
    # set_date_axes(len(s_time))
    plt.legend()
    plt.grid()
    
    
    

if __name__ == '__main__':
    y = YahooData()
    
    
    from dumbot.build_data import load_symbol_names
    
    
    symbols = load_symbol_names()
    symbol = symbols[4]
    symbol = 'AAPL'
    symbol = 'GOOG'
    df = y.get_symbol_all(symbol)
    test_smoothing(df)