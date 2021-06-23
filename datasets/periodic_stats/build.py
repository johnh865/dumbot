# -*- coding: utf-8 -*-
import pdb
import pandas as pd
import numpy as np
from functools import cached_property


from backtester.stockdata import YahooData, to_sql, SQLData, BaseData
from backtester.definitions import DF_ADJ_CLOSE
from backtester.indicators import TrailingStats, QuarterStats, MonthlyStats
from backtester.utils import extract_column

from datasets.periodic_stats.definitions import ROI_URL, ROLLING_URL


import matplotlib.pyplot as plt
import seaborn as sns
from globalcache import Cache


def sortino_ratio(r: np.ndarray, target: float, bins: int=25) -> float:
    """https://en.wikipedia.org/wiki/Sortino_ratio

    Parameters
    ----------
    r : np.ndarray
        Array of monthly/annual/etc returns.
    target : float
        Target average rate of return.
    bins : int, optional
        Number of bins for histogram. The default is 25.

    Returns
    -------
    float
        Sortino Ratio.
    """
    R = np.mean(r)
    if R == target:
        return 0.0
    
    
    
    r = r[~np.isnan(r)]     # Get rid of NAN 
    rmin = np.min(r)
    rmax = np.max(r)
    bin_arr = np.linspace(rmin, target, bins)
    if rmax > target:
        bin_arr = np.append(bin_arr, rmax)
    if rmin > target:
        return np.nan
    
    hist, edges = np.histogram(r, bins=bin_arr, density=True )
    r0arr = (edges[0:-1] + edges[1:]) / 2
    edges_delta = edges[1:] - edges[0:-1]
    
    is_below_target = r0arr <= target
    ratio = (target - r0arr)**2 * hist * is_below_target
    DR = np.sum(ratio * edges_delta)**0.5
    
    # DR = np.trapz(ratio, r0arr)**0.5
    return (R - target) / DR


class ROIStats:
    def __init__(self,
                 data: BaseData,
                 intervals=12,
                 # symbols: list[str]=(),
                 ):
        # self.symbols = symbols
        self.intervals = intervals
        self.data = data
        

    @staticmethod
    def _calc_roi_monthly(df) -> pd.Series:
        """Calculate return on investment, monthly for given dataframe."""
        s = df[DF_ADJ_CLOSE]
        t = MonthlyStats(s)
        try:
            metric = t.return_ratio
            times = t.times.astype('datetime64[M]')
            series =  pd.Series(metric, index=times, name='ROI')
            return series
            # return pd.DataFrame(series)
        except ValueError:
            d = {}
            d['ROI'] = []
            return pd.Series(d)
            # return pd.DataFrame(d)
    
        
    @cached_property
    def dict(self) -> pd.DataFrame:
        """Return on investment for YahooData."""
        # y = YahooData(self.symbols)
        y = self.data
        # def func(symbol: str):
        #     df = y.dataframes[symbol]
        #     return self._calc_roi_monthly(df)
        # m = MapData(y.symbols, func)
        new = {}
        for key, df in y.dataframes.items():
            print('calculating ROI for', key)
            new[key] = self._calc_roi_monthly(df)
        return new
        # df = pd.concat(new, axis=1, join='outer',)
        # return df
        
        
    @cached_property
    def _roi_spy_trailing_stats(self):
        series = self.dict['SPY']
        ts = TrailingStats(series, window_size=self.intervals)
        return ts
    

    def _stats_function(self, 
                        s: pd.Series, 
                        roi_spy: np.ndarray, 
                        target: float):
        """Calculate statistics for a given series, compared to SPY series."""
        delta = s.values - roi_spy
        roi_mean = s.mean()
        roi_std = s.std()
        roi_count = s.count()
        delta_mean = delta.mean()
        sharpe = delta_mean / roi_std
        sortino = sortino_ratio(s.values, target)
        
        d = {}
        d['ROI mean'] = roi_mean
        d['ROI std'] = roi_std
        d['count'] = roi_count
        d['sharpe'] = sharpe
        d['sortino'] = sortino
        return pd.Series(d)
    
    
    def _rolling_stats(self, s: pd.Series):
        
        spy = self.dict['SPY']
        d = {}
        d['SPY'] = spy
        d['AAA'] = s
        
        df = pd.concat(d, join='inner', axis=1)
        spy = df['SPY']
        s = df['AAA']
        
        ts = TrailingStats(s, window_size=self.intervals)
        ts_spy = self._roi_spy_trailing_stats
        
        value_gen = zip(ts_spy._values_intervals,
                        ts._values_intervals,
                        ts_spy.mean)
        
        outputs = []
        for value_spy, value, target in value_gen:
            
            out = self._stats_function(
                s = pd.Series(value),
                roi_spy = value_spy,
                target = target
                )
            outputs.append(out)
        
        return pd.DataFrame(outputs, index=ts.times)        

    
    @cached_property
    def trailing_stats(self) -> dict:    
        """Trailing ROI statistics."""
        d = self.dict
        new = {}
        for key, series in d.items():
            print('Getting rolling stats for', key)
            new[key] = self._rolling_stats(series)
        return new
    

def build():
    """Calculate & write return on investment statistics."""
    y = YahooData()
    stats = ROIStats(y)
    to_sql(ROI_URL, stats.dict)
    to_sql(ROLLING_URL, stats.trailing_stats)
    


def read_rolling():
    data = SQLData(ROLLING_URL)
    return data
    
def read_roi():
    data = SQLData(ROI_URL)
    return data