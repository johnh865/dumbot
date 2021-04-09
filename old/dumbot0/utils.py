# -*- coding: utf-8 -*-
import datetime
from functools import lru_cache

from sqlalchemy import create_engine
import pandas as pd
import numpy as np

import scipy
import scipy.ndimage
import scipy.stats
from scipy.stats import linregress
from scipy.ndimage import uniform_filter

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


def get_rolling_average(df: pd.DataFrame, window_size:int):
    origin = int(window_size / 2)
    times = df.index    
    values = df[DF_ADJ_CLOSE]
    smoothed = uniform_filter(
        values,
        size=window_size,
        origin=origin,
        )
    
    
    diff = np.diff(smoothed) / smoothed[1:]
    
    df2 = pd.DataFrame()
    df2[DF_ADJ_CLOSE] = values[1:]
    df2[DF_SMOOTH_CLOSE] = smoothed[1:]
    df2[DF_SMOOTH_CHANGE] = diff
    df2[DF_TRUE_CHANGE] = np.diff(values) / values[1:]
    df2.index = times[1:]
    return df2

    
def get_trading_days(date1 : datetime.date, date2 : datetime.date=None):
    """For two dates, get trading days in between."""
    dates = read_dataframe('DIS').index
    
    date1 = np.datetime64(date1)
    
    dates = dates[dates >= date1]
    
    if date2 is not None:
        date2 = np.datetime64(date2)
        dates = dates[dates <= date2]
    return dates

            
class SymbolStats:
    def __init__(self, df : pd.DataFrame, window_size : int):
        self.window_size = window_size
        self.df = df
        
        
    def trailing_rolling_avg(self):
        df = self.df
        window_size = self.window_size
        origin = int(window_size / 2)
        times = df.index    
        values = df[DF_ADJ_CLOSE]
        smoothed = uniform_filter(
            values,
            size=window_size,
            origin=origin,
            )    
        
        diff = np.diff(smoothed) / smoothed[1:]
        diff_true = np.diff(values) / values[1:]
        
        df2 = pd.DataFrame()
        df2[DF_ADJ_CLOSE] = values[1:]
        df2[DF_SMOOTH_CLOSE] = smoothed[1:]
        df2[DF_SMOOTH_CHANGE] = diff
        df2[DF_TRUE_CHANGE] = diff_true
        df2.index = times[1:]
        return df2        
    
    
    def true_rolling_avg(self):
        """Rolling average that has magical future foresight."""
        df = self.df
        window_size = self.window_size
        times = df.index    
        values = df[DF_ADJ_CLOSE]
        smoothed = uniform_filter(
            values,
            size=window_size,
            )    
        diff = np.diff(smoothed) / smoothed[1:]
        diff_true = np.diff(values) / values[1:]

        df2 = pd.DataFrame()
        df2[DF_ADJ_CLOSE] = values[1:]
        df2[DF_SMOOTH_CLOSE] = smoothed[1:]
        df2[DF_SMOOTH_CHANGE] = diff
        df2[DF_TRUE_CHANGE] = diff_true
        
        df2.index = times[1:]
        return df2      

    
    
    def linear_regression(self):
        window_size = self.window_size
        x = np.arange(window_size)
        y = self.df[DF_ADJ_CLOSE].iloc[-window_size : ]
        result = linregress(x, y)

        return result.slope, result.intercept
    
    
    def slope_normalized(self):
        m, y0 = self.linear_regression()
        y = self.df[DF_ADJ_CLOSE].iloc[-self.window_size : ]
        ymean = np.mean(y)
        return m / ymean
    
    
    def avg_slope(self):
        window_size = self.window_size
        values = df[DF_ADJ_CLOSE].iloc[-window_size-1 : ]
        diff = np.diff(values)
        return np.mean(diff)
        
        
if __name__ == '__main__':
    df = read_dataframe('VOO').iloc[-1000:]
    
    period = 21
    s = SymbolStats(df, period)
    s2 = SymbolStats(df, 81)
    
    m, y0 = s.linear_regression()
    print(m)
    print(s.avg_slope())
    import matplotlib.pyplot as plt
    
    plt.subplot(2,1,1)
    plt.plot(s.df.index, s.df[DF_ADJ_CLOSE])
    
    
    x1 = s.df.index[[-period, -1]].values
    x2 = np.array([0, period-1])
    y1 = m * (x2) + y0
    plt.plot(x1, y1)
    
    df_trailing_avg = s.trailing_rolling_avg()
    df_trailing_avg2 = s2.trailing_rolling_avg()
    df_tru_avg = s.true_rolling_avg()
    
    plt.plot(df_trailing_avg.index, df_trailing_avg[DF_SMOOTH_CLOSE], '--',
             label='trailing-21')
    
    plt.plot(df_trailing_avg2.index, df_trailing_avg2[DF_SMOOTH_CLOSE], '--',
             label='trailing-200')    
    
    # plt.plot(df_tru_avg.index, df_tru_avg[DF_SMOOTH_CLOSE],
    #          label='true')
    
    plt.grid()
    plt.legend()
    
    plt.subplot(2,1,2)
    
    plt.plot(df_trailing_avg.index, df_trailing_avg[DF_SMOOTH_CHANGE], '.--',
             label='trailing-21')
    
    plt.plot(df_trailing_avg2.index, df_trailing_avg2[DF_SMOOTH_CHANGE], '.--',
             label='trailing-200')    
    
    plt.plot(df_tru_avg.index, df_tru_avg[DF_SMOOTH_CHANGE],
              label='true')
    plt.axhline(0, color='k')
    plt.grid()
    plt.legend()
            
        