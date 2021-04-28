# -*- coding: utf-8 -*-
from numba import njit
from backtester.exceptions import NotEnoughDataError
import numpy as np
from backtester.indicators import array_windows

if __name__ == '__main__':
    
    x = np.arange(5)
    
    try:
        y = array_windows(x, 40)
    except NotEnoughDataError:
        y = 'bob'
    