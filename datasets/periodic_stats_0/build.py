"""Generate monthly periodic statistics such as Sharpe, Sortino, etc"""
# -*- coding: utf-8 -*-
# Num trading days per year = 252
import pdb
import pandas as pd
import numpy as np

from backtester.stockdata import YahooData, Indicators, MapData, LazyMap
from backtester.definitions import DF_ADJ_CLOSE
from backtester.indicators import TrailingStats, QuarterStats, MonthlyStats
from backtester.utils import extract_column

import matplotlib.pyplot as plt
import seaborn as sns
from globalcache import Cache



# %% Get stock data

# %% Define calculations

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
    R = np.mean(r)
    return (R - target) / DR


def roi(key):
    df = y_data.dataframes[key]
    s = df[DF_ADJ_CLOSE]
    t = MonthlyStats(s)
    try:
        metric = t.return_ratio
        times = t.times.astype('datetime64[M]')
        series =  pd.Series(metric, index=times, name='ROI')
        return pd.DataFrame(series)
    except ValueError:
        d = {}
        d['ROI'] = []
        return pd.DataFrame(d)
    
    
# %% Retrive Data

if __name__ == '__main__':
    cache = Cache(globals())
    y_data = YahooData()
    
    # start_date = np.datetime64('2012-01-01')
    symbols = y_data.retrieve_symbol_names()
    # y_data = y_data.filter_dates(start=start_date)
    
    # symbols = symbols[0:5]
    # symbols = ['GME']
    
    symbols.append('SPY')
    
        
    # %% Calculate ROI intervals
    
    map_data = LazyMap(symbols, roi)
    
    # map_data = MapData(symbols, roi)
    df0 = map_data['SPY']
    
    @cache.decorate
    def extract():
        df = extract_column(map_data, 'ROI')
        return df
    
    df = extract()
    
    
    # %% Calculate Sortino
    
    delta = df.values - df['SPY'].values[:, None]
    target_return = df['SPY'].mean()
    
    
    def get_sortino(df: pd.DataFrame):
        """Calcualte Sortino ratio."""
        values = df['ROI'].values
        try:
            return sortino_ratio(values, target_return)
        except ValueError:
            return np.nan
        
    sortino_map = map_data.map(get_sortino)
    sortino_stats = pd.Series(sortino_map)
    
    
    # %% Post
    
    df_delta = df.copy()
    df_delta.iloc[:,:] = delta
    
    roi_stats = df.describe()
    locs = roi_stats.loc['count'] > 4*3
    
    delta_stats = df_delta.describe()
    
    sharpe_ratio = delta_stats.loc['mean'] / roi_stats.loc['std']
    # sharpe_ratio = sharpe_ratio[locs]
    # sharpe_ratio = sharpe_ratio.sort_values()
    
    
    
    stats = {}
    stats['ROI mean'] = roi_stats.loc['mean']
    stats['ROI std'] = roi_stats.loc['std']
    stats['sharpe'] = sharpe_ratio
    stats['sortino'] = sortino_stats
    stats['# years'] = roi_stats.loc['count'] / 12
    
    stats = pd.DataFrame(stats)
    # sns.heatmap(df_delta, cmap='coolwarm')
    
    
    # %% Save data
    
    df.to_csv('monthly_ROI.csv')
    stats.to_csv('monthly_stats.csv')