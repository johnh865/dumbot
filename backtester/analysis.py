# -*- coding: utf-8 -*-
import math
import pdb


from numba import jit
import numpy as np
import pandas as pd
from functools import cached_property

class BuySell:
    """Growth & price change for all buy and sell times in a price history.

    Parameters
    ----------
    times : np.ndarray
        Time array.
    closes : np.ndarray
        Price data array.
    min_hold_days : int, optional
        Minimum days to hold. The default is 1.
    max_hold_days : int, optional
        Maximum days to hold. The default is -1. Set this so you don't 
        have to calculate as many points. Set to -1 to use entire history. 
    skip : int, optional
        Days to skip for data reduction. The default is 1.
    """ 
    def __init__(self, times, closes, 
                 min_hold_days: int=1, 
                 max_hold_days: int=-1,
                 skip: int=1):

        datas = buysell(times,
                        closes, 
                        min_hold_days,
                        max_hold_days,
                        skip)
        
        self.buy_index, self.sell_index, self.growths, self.changes = datas
        self.index = np.unique(self.buy_index)
        return
    
    
    @property
    def data(self) -> tuple[np.ndarray]:
        """All buy-sell index locations and associated growth and price change.
        
        Returns
        -------
        buy_index : np.ndarray[np.int64]
            Index location at buy time.
        sell_index : np.ndarray[np.int64]
            Index location at sell time.
        growths : np.ndarray[np.float64]
            Exponential rate of growth.
        changes : np.ndarray[np.float64]
            Relative change in price.
        """
        return (self.buy_index, self.sell_index, self.growths, self.changes)
    
    
    @cached_property
    def max_change(self) -> tuple[np.ndarray]:
        """Get maximum price change for each location within max_hold_days.
        
        Returns
        -------
        buy_index : np.ndarray[np.int64]
            Index location at buy time.
        sell_index : np.ndarray[np.int64]
            Index location at sell time.
        growths : np.ndarray[np.float64]
            Exponential rate of growth.
        changes : np.ndarray[np.float64]
            Relative change in price.

        """
        return buysell_max_change(
            self.buy_index,
            self.sell_index, 
            self.growths,
            self.changes)
        
        
    
def buysell(times, closes, 
                 min_hold_days: int=1, 
                 max_hold_days: int=-1,
                 skip: int=1):
    """Growth & price change for all buy and sell times in a price history.

    Parameters
    ----------
    times : np.ndarray
        Time array.
    closes : np.ndarray
        Price data array.
    min_hold_days : int, optional
        Minimum days to hold. The default is 1.
    max_hold_days : int, optional
        Maximum days to hold. The default is -1. Set this so you don't 
        have to calculate as many points. Set to -1 to use entire history. 
    skip : int, optional
        Days to skip for data reduction. The default is 1.

    Returns
    -------
    buy_index : np.ndarray[np.int64]
        Index location at buy time.
    sell_index : np.ndarray[np.int64]
        Index location at sell time.
    growths : np.ndarray[np.float64]
        Exponential rate of growth.
    changes : np.ndarray[np.float64]
        Relative change in price.

    """    
    
    min_hold_days = np.int64(min_hold_days)
    max_hold_days = np.int64(max_hold_days)
    skip = np.int64(skip)
    times = np.asarray(times, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    datas = _buysell(times, closes, 
                         min_hold_days=min_hold_days,
                         max_hold_days=max_hold_days,
                         skip=skip,
                         )
    return datas
    

def _buysell_nojit(
        times: np.ndarray,
        closes: np.ndarray,
        min_hold_days: int=1,
        max_hold_days: int=-1,
        skip: int=np.int64(1),):

    if skip > 1:
        original_len = len(times)
        times = times[::skip]
        closes = closes[::skip]        
    
    length = len(times)
    newlength = int(length * (length + 1) / 2)


    if max_hold_days == -1:
        max_hold_days = length

    buy_index = np.empty(newlength, dtype=np.int64)
    sell_index = np.empty(newlength, dtype=np.int64)
    growths = np.empty(newlength, dtype=np.float64)
    changes = np.empty(newlength, dtype=np.float64)
    
    kk = 0
    for ii in range(length):
        buy = closes[ii]
        jj_start = ii + min_hold_days
        jj_end = min(length, jj_start + max_hold_days)

        for jj in range(jj_start, jj_end):
            sell = closes[jj]
            time_diff = times[jj] - times[ii]
            growth = (math.log(sell) - math.log(buy)) / time_diff
            change = (sell - buy) / buy
            
            buy_index[kk] = ii
            sell_index[kk] = jj
            growths[kk] = growth
            changes[kk] = change
            # out[kk, :] = (ii, jj, growth)
            kk += 1
            
    buy_index = buy_index[0 : kk]
    sell_index = sell_index[0 : kk]
    growths = growths[0 : kk]
    changes = changes[0 : kk]
    
    # isort = np.argsort(growths)[::-1]
    # buy_index = buy_index[isort]
    # sell_index = sell_index[isort]
    # growths = growths[isort]
    # changes = changes[isort]
    
    if skip > 1: 
        buy_index = buy_index * skip
        sell_index = sell_index * skip

    return buy_index, sell_index, growths, changes

_buysell = jit(nopython=True)(_buysell_nojit)


# @jit(nopython=True)
def buysell_max_change(buy_index, sell_index, growths, changes):
    """Calculate maximum change."""
    unique_index = np.unique(buy_index)
    
    ulen = len(unique_index)
    buy_index2 = np.empty(ulen, dtype=np.int64)
    sell_index2 = np.empty(ulen, dtype=np.int64)
    growths2 = np.empty(ulen, dtype=np.float64)
    changes2 = np.empty(ulen, dtype=np.float64)    
    
    
    for ii, jj in enumerate(unique_index):
        locs = buy_index == jj
        kk = np.argmax(changes[locs])
        new_index = np.where(locs)[0][kk]
        
        buy_index2[ii] = buy_index[new_index]
        sell_index2[ii] = sell_index[new_index]
        growths2[ii] = growths[new_index]
        changes2[ii] = changes[new_index]
    
    return buy_index2, sell_index2, growths2, changes2




def avg_future_growth(times: np.ndarray,
                      closes: np.ndarray,
                      window: int) -> (np.ndarray, np.ndarray):
    
    times = np.array(times, dtype=np.float64)
    closes = np.array(closes, dtype=np.float64)
    window = np.int64(window)
    return _avg_future_growth(times, closes, window)


@jit(nopython=True)
def _avg_future_growth(times: np.ndarray, 
                       closes: np.ndarray,
                       window: int) -> (np.ndarray, np.ndarray):
    """Given a time point, average future growth rate given future window period.
    Used to meaure how good/bad it is to buy stock at a particular time point. 
    
    Parameters
    ----------
    times : np.ndarray
        Time points.
    closes : np.ndarray
        Stock price.
    window : int
        Future number of days to average.

    Returns
    -------
    np.ndarray
        For each point in `times`, the future growth rate .
    """

    len1 = len(times)
    out = np.zeros(len1, dtype=np.float64)
    for ii in range(len1 - 1):
        
        # if ii == len1-2:
        #     pdb.set_trace()
        close = closes[ii]
        time = times[ii]
        
        future_closes = closes[ii + 1   : ii + window + 1]
        future_times = times[ii + 1 : ii + window + 1]
        
        close_change = (future_closes - close) / close
        times_change = future_times - time
        growth_rate = close_change / times_change
        growth_avg = np.mean(growth_rate)
        out[ii] = growth_avg
        # out.append(growth_avg)
    # out.append(0.0)
    return times[0 : -window], out[0 : -window]
        
        





def _jit_speed_test1():
    
    # Test jit speed
    from backtester.stockdata import YahooData
    from backtester.definitions import DF_ADJ_CLOSE
    from backtester import utils
    import timeit
    
    symbol = 'VOO'
    y = YahooData([symbol])
    df = y.get_symbol_all(symbol)    
    df = df.iloc[-100::2]
    
    series = df[DF_ADJ_CLOSE]
    
    times = utils.dates2days(series.index)
    times = times.values.astype(np.float64)
    prices = series.values
    
    def test_no():
        d = _buysell_nojit(times, prices,
                            min_hold_days=np.int64(1),
                            max_hold_days=np.int64(20),
                            skip=np.int64(1),
                            )
        return d
    
    def test_jit():
        d = _buysell(times, prices,
                            min_hold_days=np.int64(1),
                            max_hold_days=np.int64(20),
                            skip=np.int64(1),
                            )
        return d
    
    t1 = timeit.timeit(test_no, number=10000)
    t2 = timeit.timeit(test_jit, number=10000)
    print('Time without jit', t1)
    print('Time using jit', t2)
    
    
    

        
        