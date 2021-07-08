# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from datasets.periodic_stats.build import ROIStats
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData


symbol = 'TSLA'
y = YahooData(['SPY', symbol])
stats = ROIStats(y)
d = stats.dict
ts = stats.trailing_stats

close = y.dataframes[symbol][DF_ADJ_CLOSE]
sharpe = ts[symbol]['sharpe']
sortino = ts[symbol]['sortino']


xlim1 = np.datetime64('2006-01-01')
xlim2 = np.datetime64('2022-01-01')
plt.subplot(2,1,1)
plt.semilogy(close, label='close')
plt.grid()
plt.legend()
plt.xlim(xlim1, xlim2)

plt.subplot(2,1,2)
plt.plot(sharpe, label='avg sharpe')
plt.plot(sortino, label='avg sortino')
plt.grid()
plt.legend()
plt.xlim(xlim1, xlim2)
