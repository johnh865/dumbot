# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from backtester.stockdata import DictData
from datasets.periodic_stats.build import ROIStats

t1 = np.datetime64('2020-01-01')
t2 = np.datetime64('2020-05-01')
times = np.arange(t1, t2)
time_days = (times - times[0]).astype('timedelta64[D]').astype(float)
values = time_days * 
times = pd.DatetimeIndex(times)



series = pd.Series(times.month, index=times, name='month')

df = series.to_frame()
data = DictData({'A' : df})

r = ROIStats(data)
