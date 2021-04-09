# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
from dumbot import bot
import numpy as np

def test_smooth():
    s = bot.StockData('VOO')
    window_size = 21
    
    data1 = s.get_range(window_size)
    data_smooth = s.smooth_range(21, window_size=window_size)
    
    average = np.mean(data1[-window_size:])
    assert average == data_smooth['Close'][-1]
    
    
def plot_smooth():
    s = bot.StockData('DIS')
    days = 600
    data1 = s.get_range(days=days)
    data2 = s.smooth_range(days=days, window_size=15, order=1)
    
    plt.figure()
    
    plt.subplot(2,1,1)
    plt.plot(data1.index, data1.values)
    plt.plot(data2.index, data2['Close'])
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))   
    plt.grid(which='minor')
    
    plt.subplot(2,1,2)
    plt.plot(data2.index, data2['Relative_Change'])
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))   
    plt.axhline(0, color='k')
    plt.grid(which='minor')

    
if __name__ == '__main__':
    test_smooth()
    plot_smooth()
