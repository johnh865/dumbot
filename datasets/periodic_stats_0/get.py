# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datasets.periodic_stats.definitions import (
    FILE_ROI, FILE_STATS, FILE_ROLLING_STATS)
from datasets.periodic_stats.build import sortino_ratio
import matplotlib.pyplot as plt

STATS = ['ROI mean',
         'ROI std',
         'sharpe',
         'sortino',
         '# years',
         ]
# df_stats = pd.read_csv(FILE_STATS, index_col=0)


def read_roi():
    df_roi = pd.read_csv(FILE_ROI, index_col=0)
    df_roi.index = pd.to_datetime(df_roi.index)
    return df_roi


def calculate_stats(df: pd.DataFrame):
    
    roi_spy = df['SPY']
    target = np.mean(roi_spy)
    
    
    # Calculate difference between SPY and other returns
    delta = df.values - df['SPY'].values[:, None]
    df_delta = df.copy()
    df_delta.iloc[:,:] = delta    
        
    
    # roi_stats = df.describe()    
    roi_mean = df.mean()
    roi_std = df.std()
    roi_count = df.count()
    
    # delta_stats = df_delta.describe()
    delta_mean = df_delta.mean()

    # Calculate Sortino
    sortino_map = {}
    for key in df.columns:
        roi = df[key]
        sortino_map[key] = sortino_ratio(roi, target)        
    sortino_stats = pd.Series(sortino_map)
    
    # Calculate Sharpe
    sharpe_ratio = delta_mean / roi_std

    stats = {}
    stats['ROI mean'] = roi_mean
    stats['ROI std'] = roi_std
    stats['sharpe'] = sharpe_ratio
    stats['sortino'] = sortino_stats
    stats['# years'] = roi_count / 12
    
    stats = pd.DataFrame(stats)
    return stats


def save_rolling_stats():
    df_roi = read_roi()
    dates = df_roi.index
    span_months = 36
    new = {}
    
    for ii, date in enumerate(dates[span_months :]):
        print(ii, date)
        df_i = df_roi.iloc[ii : ii + span_months]
    
        stats_i = calculate_stats(df_i).unstack()
        
        # date_i = df_roi.index[ii + span_months - 1] + pd.DateOffset(months=1)
        date_i = date + pd.DateOffset(months=1)
        new[date_i] = stats_i    
    
    df_new = pd.DataFrame(new).T
    df_new.to_csv(FILE_ROLLING_STATS)
    return df_new

    
def read_rolling_stats():
    df = pd.read_csv(FILE_ROLLING_STATS,
                     header=[0,1],
                     index_col=0
                     )
    df.index = pd.to_datetime(df.index)
    
    new = {}
    for key in STATS:
        new[key] = df[key]
    return new

if __name__ == '__main__':

    df = save_rolling_stats()
    df = read_rolling_stats()
    
    
    df1 = df['sharpe']
    stocks = df1.keys().values
    np.random.shuffle(stocks)
    
    name1 = stocks[0]
    name2 = stocks[1]
    plt.plot(df1[name1], label=name1)
    plt.plot(df1[name2], label=name2)
    plt.grid()
    plt.legend()