# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,4,1000)
y = np.sin(2*np.pi*x/4)

dydx = np.gradient(y) / np.gradient(x)
d2ydx2 = np.gradient(dydx) / np.gradient(x)


x1 = np.arange(0, 20)
y1 = np.sin(2*np.pi*x1/16)
dy1dx = np.gradient(y1) / np.gradient(x1)
d2y1dx2 = np.gradient(dy1dx) / np.gradient(x1)

plt.plot(x,y, label='y')
plt.plot(x, dydx, label='dy/dx')
plt.plot(x, d2ydx2, label='d2y/dx2')

plt.plot(x1,y1, '--', label='y1')
plt.plot(x1, dy1dx, '--', label='dy1/dx')


plt.grid()
plt.legend()