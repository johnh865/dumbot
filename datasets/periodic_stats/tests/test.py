# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester.stockdata import DictData
from datasets.periodic_stats.build import ROIStats


def build_data():
    t1 = np.datetime64('2020-01-01')
    t2 = np.datetime64('2022-05-01')

    dates = np.arange(t1, t2)
    time_days = (dates - dates[0]).astype('timedelta64[D]').astype(float)
    dates = pd.DatetimeIndex(dates)
        
    y = 0
    new = []
    for time, date in zip(time_days, dates):
        month = date.month
        new.append(y)
        y = y + month
        
    new = np.array(new)
    return pd.Series(new, index=dates, name='Adj Close')



series = build_data()
plt.plot(series, '.-')
plt.grid()
df = series.to_frame()
data = DictData({'A' : df, 
                 'SPY' : df})

r = ROIStats(data)

roi_data = r.data.dataframes['A']
trailing_data = r.trailing_stats
