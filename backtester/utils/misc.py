# -*- coding: utf-8 -*-
from typing import Sequence
from numbers import Number

import numpy as np
import pandas as pd
from numba import njit, TypingError
    
def delete_attr(obj, name):
    """Try to delete attribute. Used for cleaning up cached_property."""
    try:
        delattr(obj, name)
    except AttributeError:
        pass
    

def cross(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` and `series2` just crossed (either
    direction).

        >>> cross(self.data.Close, self.sma)
        True

    """
    return crossover(series1, series2) or crossover(series2, series1)


def crossover(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` just crossed over
    `series2`.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    series1 = (
        series1.values if isinstance(series1, pd.Series) else
        (series1, series1) if isinstance(series1, Number) else
        series1)
    series2 = (
        series2.values if isinstance(series2, pd.Series) else
        (series2, series2) if isinstance(series2, Number) else
        series2)
    try:
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]
    except IndexError:
        return False


class InterpConstAfter:
    """Interpolate data of form y' = f(x'), assuming constant staggered 
    'staircase' behavior of y'. y' runs constant as x' increases until 
    x' reaches point (x[i], y[i]), at which y' changes to y[i]. 
    
    Parameters
    ----------
    x : np.ndarray
        indendent x values. Must be sorted. 
    y : np.ndarray
        dependent y values.
    x0 : np.ndarray or numeric
        x locations to interpolate.

    Returns
    -------
    y0 : np.ndarray
        Interpolated values.    """
    
    def __init__(self, x, y, before=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        if before is not None:
            self._ymin = before
        else:
            self._ymin = self.y[0]
        
    
    def scalar(self, x0: float) -> float:
        """Interpolate for scalar input."""
        return _interp_const_njit_scalar(self.x, self.y, x0, self._ymin)
    
    
    def array(self, x0: np.ndarray) -> np.ndarray:
        """Interpolate for array input."""
        return  _interp_const_njit_array(self.x, self.y, x0, self._ymin)
    
    
    def __call__(self, x0):
        """If you don't know type you can just call but this is slower.
        Use self.array or self.scalar for faster calculation."""
        if np.isscalar(x0):
            return self.scalar(x0)
        else:
            return self.array(x0)
        

@njit      
def _interp_const_njit_array(x, y, x0, before):
    sort_index = np.searchsorted(x, x0, side='right') - 1
    y0 = y[sort_index]
    locs = x0 < x[0]
    y0[locs] = before    
    return y0


@njit      
def _interp_const_njit_scalar(x, y, x0, before):
    if x0 < x[0]:
        return before
    
    sort_index = np.searchsorted(x, x0, side='right') - 1
    y0 = y[sort_index]
    return y0



"""Just storing old methods here for now... Keep for speed comparison?"""
@njit      
def _interp_const_after_njit(x, y, x0, before=None):
    imax = len(x) - 1
    
    # Get x0 locations that exactly match the date        
    sort_index = np.searchsorted(x, x0)
    sort_index = np.minimum(sort_index, imax)
    y0 = y[sort_index]        
    
    # Get x0 locations that do not exactly match date. 
    # Adjust their index by -1.
    # Exclude x0 locations greater than x.max()    
    adjust_locs= (x[sort_index] != x0) & (x0 < x[-1])
    adjust_index = sort_index[adjust_locs] - 1
    adjust_index = np.maximum(adjust_index, 0)    
    y0[adjust_locs] = y[adjust_index]
    
    # Set values for before x.min()
    if before is not None:
        locs = x0 < x[0]
        y0[locs] = before
    return y0


def _interp_const_after_old1(x, y, x0, before=None):
    """Old verion, don't use."""
    x = np.asarray(x)
    y = np.asarray(y)
    x0_is_scalar = np.isscalar(x0)
    x0 = np.atleast_1d(x0)
    y0 = _interp_const_after_njit(x, y, x0, before=before)
    if x0_is_scalar:
        y0 = y0[0]
    return y0



interp_const_after = _interp_const_after_old1

def _interp_const_after_old2(x: np.ndarray,
                       y: np.ndarray,
                       x0: np.ndarray,
                       before=None) -> np.ndarray:
    """Interpolate data of form y' = f(x'), assuming constant staggered 
    'staircase' behavior of y'. y' runs constant as x' increases until 
    x' reaches point (x[i], y[i]), at which y' changes to y[i]. 
    
    Parameters
    ----------
    x : np.ndarray
        indendent x values. Must be sorted. 
    y : np.ndarray
        dependent y values.
    x0 : np.ndarray or numeric
        x locations to interpolate.

    Returns
    -------
    y0 : np.ndarray
        Interpolated values.
    """
    
    
    y = np.asarray(y)
    x = np.asarray(x)
    
    # Check to see if input x0 is an array. If it is not, convert back to scalar
    # At the end. 
    x0_not_arr = np.isscalar(x0)
    
    x0 = np.atleast_1d(x0)
    imax = len(x) - 1
    
    # Get x0 locations that exactly match the date    
    sort_index = np.searchsorted(x, x0)
    sort_index = np.minimum(sort_index, imax)
    
    
    y0 = y[sort_index]
    
    # Get x0 locations that do not exactly match date. 
    # Adjust their index by -1.
    # Exclude x0 locations greater than x.max()
    adjust_locs= (x[sort_index] != x0) & (x0 < x[-1])
    
    adjust_index = sort_index[adjust_locs] - 1
    adjust_index = np.maximum(adjust_index, 0)
    
    y0[adjust_locs] = y[adjust_index]
    
    # Set values for before x.min()
    if before is not None:
        locs = x0 < x[0]
        y0[locs] = before
    
    if x0_not_arr:
        y0 = y0[0]
        
        
    return y0

    
