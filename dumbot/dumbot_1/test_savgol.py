# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


x = np.linspace(0,4,1000)
y = np.sin(2*np.pi*x/4)
noise = np.random.normal(size=x.shape)
y = y + noise

window = 99
f1 = savgol_filter(y, window, 3)

x2 = x[-window:]
f2 = savgol_filter(y[-window:], window, 3)
assert f2[-1] == f1[-1]


plt.plot(x, y, label='true', alpha=.2)
plt.plot(x, f1, label='filter-all')
plt.plot(x2, f2, '--', label='filter-100')
plt.legend()