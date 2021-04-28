# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cwt, ricker


x = np.arange(1000)
y = np.sin(2*np.pi*x / 100) +  np.sin(2*np.pi*x / 50)

rs = np.random.RandomState(0)
r = rs.normal(size=x.shape) * .5
y = y + r

f = np.fft.fft(y)



y2 = np.fft.ifft(f)

plt.subplot(3,1,1)
plt.plot(x,y, '.-')

plt.subplot(3,1,2)
plt.plot(np.abs(f))
plt.xlim(0, 500)

