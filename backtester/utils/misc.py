# -*- coding: utf-8 -*-
from typing import Sequence
from numbers import Number

import numpy as np
import pandas as pd

    
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




def interp_const_after(x: np.ndarray,
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

    
    # len1 = len(x)
    # len0 = len(x0)
    # insert_index = np.searchsorted(x, x0)
    # new_index = np.arange(len0)
    
    # exact_index = insert_index[dates[insert_index] == x0]
    # new = values[insert_index]
    # n
    
    
    
    
    
    
    
    # old_locs = np.arange(len1, dtype=int)
    # new_locs = np.arange(len0, dtype=int)
    
    # new = np.empty(len0, dtype=values.dtype)
    
    # exact_index = insert_index[dates[insert_index] == dates0]
    # new[new_locs[exact_index]] = dates0
    
    # between_index = ~exact_index
    # new[between_index] = 
    
    
    



    # def get_available_funds(self, date: datetime.datetime):
    #     """Get available cash funds for a given date."""
    #     balances = self.balances
    #     balance1 : AccountBalance
    #     balance2 : AccountBalance
        
    #     if len(self.balances) == 0:
    #         return self.init_funds 
     
    #     if date < balances[0].date:
    #         return self.init_funds 
        
    #     if len(balances) == 1:
    #         return balances[0].available_funds
        
    #     for ii in range(len(balances)-1):
    #         balance1 = balances[ii]
    #         balance2 = balances[ii + 1]
    #         if balance1.date <= date < balance2.date:
    #             return balance1.available_funds
        
    #     # If date has gone through whole history, return last share value.
    #     return balance2.available_funds    
