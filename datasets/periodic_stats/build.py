# -*- coding: utf-8 -*-
import pdb
import pandas as pd
import numpy as np
from functools import cached_property
from sqlalchemy.exc import OperationalError

from backtester.stockdata import (YahooData, to_sql, to_parquet,
                                  SQLData, BaseData, ParquetData
                                  )
from backtester.definitions import DF_ADJ_CLOSE
from backtester.indicators import TrailingStats, QuarterStats, MonthlyStats
from backtester.utils import extract_column

from datasets.periodic_stats.definitions import ROI_URL, ROLLING_URL
from datasets.periodic_stats.definitions import ROI_DIR, ROLLING_DIR


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


class ROI:
    """Calculate or read Return on Investment data"""
    def __init__(self, data: BaseData=None):
        if data is None:
            self.data = YahooData()
            try:
                self.read()
            except FileNotFoundError:
                self.save()
    
        
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
        
        
    @cached_property
    def calculate(self) -> dict[pd.Series]:
        """Return on investment (ROI) for YahooData."""
        y = self.data
        new = {}
        for key, df in y.dataframes.items():
            print('calculating ROI for', key)
            new[key] = self._calc_roi_monthly(df)
        return new    
    
    
    @staticmethod
    def read() -> ParquetData:
        """Read ROI data from SQL."""
        data = ParquetData(ROI_DIR)
        return data    
    
    
    @classmethod
    def from_yahoo(cls):
        data = YahooData()
        return ROI(data)
    
    
    def save(self, name=''):
        if name == '':
            name = ROI_DIR
        # to_sql(ROI_URL, self.calculate)
        to_parquet(ROI_DIR, self.calculate)
        
    
    

class ROIStats:
    def __init__(self,
                 # data: BaseData,
                 intervals=12,
                 # symbols: list[str]=(),
                 ):
        # self.symbols = symbols
        self.intervals = intervals
        # self.data = data
        

    # @staticmethod
    # def _calc_roi_monthly(df) -> pd.Series:
    #     """Calculate return on investment, monthly for given dataframe."""
    #     s = df[DF_ADJ_CLOSE]
    #     t = MonthlyStats(s)
    #     try:
    #         metric = t.return_ratio
    #         times = t.times.astype('datetime64[M]')
    #         series =  pd.Series(metric, index=times, name='ROI')
    #         return series
    #         # return pd.DataFrame(series)
    #     except ValueError:
    #         d = {}
    #         d['ROI'] = []
    #         return pd.Series(d)
    #         # return pd.DataFrame(d)
    
        
    # @cached_property
    # def dict(self) -> pd.DataFrame:
    #     """Return on investment (ROI) for YahooData."""
    #     # y = YahooData(self.symbols)
    #     y = self.data
    #     # def func(symbol: str):
    #     #     df = y.dataframes[symbol]
    #     #     return self._calc_roi_monthly(df)
    #     # m = MapData(y.symbols, func)
    #     new = {}
    #     for key, df in y.dataframes.items():
    #         print('calculating ROI for', key)
    #         new[key] = self._calc_roi_monthly(df)
    #     return new
    #     # df = pd.concat(new, axis=1, join='outer',)
    #     # return df
    
    
    @cached_property
    def dict(self):
        return ROI.read().dataframes
        
        
    # @cached_property
    # def _roi_spy_trailing_stats(self):
    #     """Get trailing statistics for SPY"""
    #     series = self.dict['SPY']
    #     ts = TrailingStats(series, window_size=self.intervals)
    #     return ts
    

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
        """Calculate Rolling statistics."""
        
        # Get periodic ROI for SPY
        spy = self.dict['SPY']
        d = {}
        d['SPY'] = spy
        d['AAA'] = s
        
        # Concat to make sure the indices for SPY & the symbol are the same. 
        df = pd.concat(d, join='inner', axis=1)
        spy = df['SPY']['ROI']
        s = df['AAA']['ROI']
        
        # Construct Trailing statistics for periodic ROI
        ts = TrailingStats(s, window_size=self.intervals)
        # ts_spy = self._roi_spy_trailing_stats
        ts_spy = TrailingStats(spy, window_size=self.intervals)
        
        # Get generator for SPY & stock symbol values as well as mean SPY ROI
        # Loop through all intervals to perform calculations.  
        value_gen = zip(ts_spy._values_intervals,
                        ts._values_intervals,
                        ts_spy.mean)
        
        outputs = []
        for value_spy, value, target in value_gen:
            
            out = self._stats_function(
                s = pd.Series(value),
                roi_spy = value_spy,
                target = target,
                )
            outputs.append(out)
        
        # Modify times so that display time is 1 period ahead. 
        times_new = pd.DatetimeIndex(ts.times) + pd.DateOffset(months=1)
        
        return pd.DataFrame(outputs, index=times_new)        

    
    @cached_property
    def trailing_stats(self) -> dict:    
        """Trailing ROI statistics."""
        d = self.dict
        new = {}
        for key, series in d.items():
            print('Calculating rolling stats for', key)
            new[key] = self._rolling_stats(series)
        return new
    
    
    def save(self):
        symbol_prefix=f'symbol-{self.intervals}-'
        # to_sql(ROLLING_URL, self.trailing_stats, symbol_prefix=symbol_prefix)
        to_parquet(ROLLING_DIR, self.trailing_stats, symbol_prefix=symbol_prefix)
        
    
    def read(self) -> ParquetData:
        """Read ROI data from SQL."""
        symbol_prefix=f'symbol-{self.intervals}-'
        data = ParquetData(ROLLING_DIR, prefix=symbol_prefix)
        
        try:
            key = next(iter(data.dataframes.keys()))
            data.dataframes[key]
        
        except OperationalError:
            self.save()
            data = ParquetData(ROLLING_DIR, prefix=symbol_prefix)
        return data            
        
        
def build_roi():
    ROI.from_yahoo().save()
    

# def build():
#     """Calculate & write return on investment statistics."""
#     y = YahooData()
#     stats = ROIStats(y, intervals=24)
#     to_sql(ROI_URL, stats.dict)
#     to_sql(ROLLING_URL, stats.trailing_stats)
    

# def test():
#     y = YahooData(['SPY', 'GOOG'])
#     stats = ROIStats(y)
#     d = stats.dict
#     ts = stats.trailing_stats
#     # pdb.set_trace()
#     return



# def read_rolling():
#     data = SQLData(ROLLING_URL)
#     return data
    
# def read_roi():
#     data = SQLData(ROI_URL)
#     return data

# r = ROIStats(intervals=36)
# r.save()
if __name__ == '__main__':
    # y = YahooData(['SPY', 'GOOG'])
    roi = ROI()
    r = ROIStats(36)
    r.save()
    r.read()