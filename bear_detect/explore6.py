# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

from backtester.indicators import TrailingStats
from backtester.smooth import TrailingSavGol

from backtester.stockdata import YahooData, Indicators, TableData
from backtester.definitions import DF_ADJ_CLOSE
from scipy.signal import find_peaks

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
    xmean = np.mean(x)
    ymean = np.mean(y)        
    x1 = x - xmean
    y1 = y - ymean
    ss_xx = np.sum(x1**2)
    ss_xy = np.sum(x1 * y1)
    ss_yy = np.sum(y1**2)
    
    m = ss_xy / ss_xx
    b = ymean - m * xmean
    r = ss_xy / np.sqrt(ss_xx * ss_yy)
    return m, b, r
    


def trough_pivot(x: np.ndarray):
    imin = np.argmin(x)
    
    
    left = x[0 : imin]
    right = x[imin :]
    
    left_imax = np.argmax(left)
    right_imax = np.argmax(right) + imin
    
    
    
    imax = np.argmax(x)
    xmin = x[imin]
    xmax = x[imax]
    
    
