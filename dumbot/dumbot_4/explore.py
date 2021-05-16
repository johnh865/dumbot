# -*- coding: utf-8 -*-
import numpy as np
from backtester.stockdata import YahooData, Indicators
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE

import matplotlib.pyplot as plt

def my_return(df):
    s = df[DF_ADJ_CLOSE]
    size = 252
    ts = TrailingStats(s, window_size=size)
    return ts.return_ratio * 252 / size


y = YahooData()


date1 = np.datetime64('2021-05-01')
date0 = date1 - np.timedelta64(6*365, 'D')



indicator = Indicators(y)
indicator.create(my_return, name='return(Yr)')
df = indicator.extract_column('return(Yr)')

ii = (df.index.values >= date0) & (df.index.values <= date1)
df2 = df.iloc[ii]
descr = df2.describe().sort_values(by='50%', axis=1, ascending=False)


# %%
plt.plot(y.dataframes['ALGN'][DF_ADJ_CLOSE])
ax = plt.gca()
ax.set_yscale('log')