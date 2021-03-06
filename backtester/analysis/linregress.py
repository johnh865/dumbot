# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtester.indicators import array_windows, append_nan



def regression(x: np.ndarray, y: np.ndarray):
    """Regression on multiple rows of data in 2-dimensional array.
        
    https://mathworld.wolfram.com/LeastSquaresFitting.html    

    Parameters
    ----------
    x : np.ndarray of shape (a, b)
        x-data to be fit across axis 1 for each row.
    y : np.ndarray of shape (a, b)
        y-data to be fit across axis 1 for each row.

    Returns
    -------
    m : np.ndarray of shape (a,)
        Regression slope for each row.
    b : np.ndarray of shape (a,)
        Regression intercept for each row.
    r : np.ndarray of shape (a,)
        Regression Pearson's correlation coefficient for each row.
    """
    x = np.asarray(x)
    xmean = np.mean(x, axis=1)
    ymean = np.mean(y, axis=1)        
    x1 = x - xmean[:, None]
    y1 = y - ymean[:, None]
    ss_xx = np.sum(x1**2, axis=1)
    ss_xy = np.sum(x1 * y1, axis=1)
    ss_yy = np.sum(y1**2, axis=1)
    
    m = ss_xy / ss_xx
    b = ymean - m * xmean
    r = ss_xy / np.sqrt(ss_xx * ss_yy)
    return m, b, r
    

def regression_series(sx: pd.Series, sy: pd.Series, window: int):
    new = [sx, sy]
    
    df = pd.concat(new, axis=1, join='outer',)
    values = df.values
    index = df.index.values
    x = values[:, 0]
    y = values[:, 1]
    x1 = array_windows(x, window=window)
    y1 = array_windows(y, window=window)
    m, b, r = regression(x1, y1)
    
    m = append_nan(m, window)
    b = append_nan(b, window)
    r = append_nan(r, window)
    return m, b, r, index


class RegressionSeries:
    def __init__(self, sx: pd.Series, sy: pd.Series, window: int):
        
        new = [sx, sy]
        df = pd.concat(new, axis=1, join='outer',)
        values = df.values
        index = df.index.values
        x = values[:, 0]
        y = values[:, 1]

        
        x1 = array_windows(x, window=window)
        y1 = array_windows(y, window=window)
        m, b, r = regression(x1, y1)
        
        m = append_nan(m, window)
        b = append_nan(b, window)
        r = append_nan(r, window)      
        
        self.index = index
        self.x = x
        self.y = y
        self.m = m
        self.b = b
        self.r = r
    
        
        

from backtester.stockdata import YahooData
from backtester.definitions import DF_ADJ_CLOSE

yahoo = YahooData()
symbols = yahoo.symbols.copy()
np.random.shuffle(symbols)


symbol1 = symbols[0]
symbol2 = symbols[1]
symbol_base = 'SPY'


s1 = yahoo.dataframes[symbol1][DF_ADJ_CLOSE]
s2 = yahoo.dataframes[symbol2][DF_ADJ_CLOSE]
s3 = yahoo.dataframes[symbol_base][DF_ADJ_CLOSE]

r1 = RegressionSeries(s3, s1, 60)
r2 = RegressionSeries(s3, s2, 60)

plt.figure()

plt.subplot(2,2,1,)
plt.title('Correlation Coeff vs Time')
plt.plot(r1.index, r1.r, label=f'{symbol1}-{symbol_base}')
plt.plot(r2.index, r2.r, label=f'{symbol2}-{symbol_base}')
plt.legend()
plt.grid()

plt.subplot(2,2,3)
plt.plot(r1.x, r1.y, '.', alpha=.2)
plt.xlabel(symbol_base)
plt.ylabel(symbol1)
plt.grid()


plt.subplot(2,2,4)
plt.plot(r2.x, r2.y, '.', alpha=.2)
plt.xlabel(symbol_base)
plt.ylabel(symbol2)
plt.grid()


ax = plt.subplot(2,2,2)
plt.plot(r1.index, r1.x, label=symbol_base)
plt.plot(r1.index, r1.y, label=symbol1)
plt.plot(r2.index, r2.y, label=symbol2)
plt.legend()
plt.grid(which='both')

ax.set_yscale('log')
ax.set_ylim(1, None)
