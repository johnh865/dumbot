# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt


from backtester.stockdata import YahooData
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE
yahoo = YahooData()
df = yahoo['GOOG']
series = df[DF_ADJ_CLOSE]
y = series.values
logy = np.log(y)

windows = [20, 50, 70, 90, 600]
growths = []
for window in windows:
    ts = TrailingStats(series, window)
    growths.append(ts.exp_growth)

    

date1 = np.datetime64('2008-01-01')
date2 = np.datetime64('2021-01-01')

# f = np.abs(np.fft.fft(logy))
ax = plt.subplot(2,1,1)
plt.plot(series)
ax.set_yscale('log')
plt.grid()
plt.xlim(date1, date2)


plt.subplot(2,1,2)
for ii, window in enumerate(windows):
    plt.plot(series.index, growths[ii], label=window)
plt.legend()
plt.grid()
plt.xlim(date1, date2)

