# -*- coding: utf-8 -*-

from backtester.utils.dates import round_to_quarters, get_quarters
import numpy as np
import pandas as pd

def check_array_equals(a, b):
    assert np.all(a == b)
    


date1 = np.datetime64('1995-04-23')
date2 = np.datetime64('2021-06-04')
dates = np.arange(date1, date2, 100)

q = round_to_quarters(dates)


series = pd.DatetimeIndex(dates)
seriesq = pd.DatetimeIndex(q)

quarters1 = get_quarters(dates)
quarters2 = series.quarter
quarters3 = seriesq.quarter


check_array_equals(quarters1, quarters2)
check_array_equals(quarters1, quarters3)
check_array_equals(series.year, seriesq.year)