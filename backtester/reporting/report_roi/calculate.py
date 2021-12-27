# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pdb 
import matplotlib.pyplot as plt
import os
from os.path import dirname, join, exists
from functools import cached_property
import pickle


from backtester.stockdata import YahooData2, TableData, BaseData
from backtester.roi import ROI, annualize
from backtester.definitions import DF_ADJ_CLOSE
data = YahooData2()
data.client.update()

start = np.datetime64('2001-01-01')
data = data.filter_dates(start=start)

intervals = [7, 30, 365//4,]
_filepath = __file__
_dirpath = dirname(_filepath)


class PostData:
    def __init__(self, path:str):
        self.path = path
        self.data
        if exists(path):
            with open(path, 'rb') as f:
                self.roi = pickle.load(f)
        else:
            self.roi

    
    
    @cached_property
    def data(self):
        data1 = YahooData2()
        data1.client.update()
        
        start = np.datetime64('2001-01-01')
        data1 = data1.filter_dates(start=start)
        return data1
    
    
    @cached_property
    def roi(self) -> BaseData:
        return self.data.tmap(get_roi)
    
    
    def roi_between(self, start: np.datetime64, end: np.datetime64):
        func = lambda x : roi_stats(x, start, end)
        # out = self.roi.tmap(func)
        
        out  = {k : func(v) for (k, v) in self.roi.items()}
        # map(func, )
        out = pd.DataFrame(out).T
        return out
    
    
    def save(self):
        roi = dict(self.roi)
        with open(self.path, 'wb') as f:
            pickle.dump(roi, f)
            
            
    def update(self):
        os.remove(self.path)
        del self.roi
        del self.data
        



def get_roi(table: TableData) -> dict[str, pd.Series]:
    """Calculate Return on Investments for available dates."""
    closes = table.dataframe['Adj Close']
    out = {}
    for interval in intervals:
        roi = ROI(closes, interval)
        times = roi.times_end
        daily = roi.daily_adjusted
        series = pd.Series(daily, index=times)
        out[interval] = series
    return out


def roi_stats(data: dict[str, np.ndarray], start: np.datetime64, end: np.datetime64):
    
    new = {}
    
    for interval, roi in data.items():
        values = roi.values
        i1, i2 = np.searchsorted(roi.index, [start, end])
        values = values[i1 : i2+1]
        
        new[f'avg {interval}'] = values.mean()
        new[f'std {interval}'] = values.std()
        new[f'len {interval}'] = len(values)
    return new


# def post(table: TableData, start: np.datetime64, stop: np.datetime64):
#     closes = table.iloc((:, 'Adj Close'))
#     closes = closes.array_between(start, stop)
#     series = pd.Series(closes, index=table.times)
    
#     d = {}
#     for interval in intervals:
#         roi = ROI(closes, interval)
        



# def post(df: pd.DataFrame):
#     """Get daily return on investment statistics."""
#     # df = tabledata.dataframe
#     closes = df[DF_ADJ_CLOSE]
    
#     d = {}
#     for interval in intervals:
#         roi = ROI(closes, interval)
#         times = roi.times_end
#         daily = roi.daily_adjusted
#         series = pd.Series(daily, index=times)
        
#         d['ROI ' + str(interval)] = series
#         # d['avg ' + str(interval)] = series.mean()
#         # d['std ' + str(interval)] = series.std()
#         # d['len ' + str(interval)] = len(series)
#         # d['start_date'] = series.index.min()
#         # d['last_date'] = series.index.max()
    
#     # d['len'] = len(closes)
        
    
#     return d


# data2 = {}
# for ii, key in enumerate(data.dataframes):
#     df = post(data[key])
#     df.name = key
#     data2[key] = df

# df = pd.DataFrame(data2).T

###########################################################################
# Plotting


# Remove data without much data.
# df = df.loc[df['len 7'] > 50]

# plt.plot(df['std 30'], df['avg 30'], '.')


# df.to_pickle(join(_dirpath, 'dataframe_output.pkl'))

path = join(_dirpath, 'roi_output.pkl')
postdata = PostData(path)
postdata.save()
test = postdata.roi_between('2011-01-01', '2021-02-01')

# data.to_pickle()