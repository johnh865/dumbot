"""Compare filter performance of exponential regression VS Savitzy-Golay.
Turns out savitzy-golay can be a lot better.

Still is pretty bad for predicting the future.
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import backtester

from backtester.indicators import TrailingStats
from backtester.analysis import BuySell, avg_future_growth
from backtester.stockdata import YahooData
from backtester.definitions import DF_ADJ_CLOSE
from backtester import utils
from backtester.smooth import SmoothOptimize, TrailingSavGol


y = YahooData()
names = y.get_symbol_names()
np.random.seed(7)
np.random.shuffle(names)

symbol = names[7]
symbol = 'MSFT'
date1 = np.datetime64('2007-01-01')
date2 = np.datetime64('2012-08-01')
df = y.get_symbol_before(symbol, date2)
# df = y['GOOG']
ii = df.index.values >= date1




7
times = df.index
close = df[DF_ADJ_CLOSE]
log_close = np.log(close)


error_amp = 1
signal_period = 100
max_accel = (np.pi / signal_period)**2 * error_amp
opt = SmoothOptimize(log_close, error_amp=error_amp, max_accel=max_accel)

print('Optimal window = ', opt.opt_window)

ts = TrailingStats(close, 100)
tsg = TrailingSavGol(log_close, 101)


c2 = np.log(ts.exp_reg_value)
dc2 = ts.exp_growth

plt.subplot(2,1,1)
plt.title(symbol)
plt.plot(times[ii], log_close[ii], label='true')
plt.plot(times[ii], opt.position[ii], label='smooth')
plt.plot(times[ii], c2[ii], label='exp-smooth', alpha=.2)
plt.plot(times[ii], tsg.position[ii], label='sg-smooth-trailing', alpha=.5)

plt.grid()
plt.legend()


p = opt.position
dpdt = np.gradient(p) / np.gradient(ts.time_days_int)



plt.subplot(2,1,2)
plt.title('Velocity Estimation')
plt.plot(times[ii], np.zeros(times[ii].shape))
plt.plot(times[ii], opt.speed[ii], label='smooth')
plt.plot(times[ii], dc2[ii], label='exp-smooth', alpha=.2 )
plt.plot(times[ii], tsg.velocity[ii], label='sg-smooth-trailing', alpha=.5)


# plt.plot(times[ii], dpdt[ii], label='smooth-2')
plt.grid()
plt.legend()

# plt.plot(opt.result[0], opt.result[1])
# plt.grid()

plt.figure()
plt.plot(tsg.velocity[ii], opt.speed[ii] , '.', alpha=.2)
plt.grid()
plt.xlabel('Predicted velocity')
plt.ylabel('Actual velocity')



# Find negative speeds
trailing_locs = tsg.velocity[ii] < 0
smoothed_locs = opt.speed[ii] < 0

matching_locs = trailing_locs == smoothed_locs
num = np.sum(matching_locs) / len(trailing_locs)
