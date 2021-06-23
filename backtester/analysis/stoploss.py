# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtester.indicators import TrailingStats
from backtester.stockdata import YahooData
from backtester.definitions import DF_ADJ_CLOSE
import matplotlib.pyplot as plt


class StopLoss:
    def __init__(self, series: pd.Series, window=150,
                 stop=0.05,
                 restart=0.10):
        self.series = series
        ts = TrailingStats(series, window_size=window)
        self.ts = ts
        self.loss = ts.max_loss
    
    

y = YahooData()
s = y.dataframes['SPY'][DF_ADJ_CLOSE]

stoploss = StopLoss(s)


plt.plot(stoploss.ts.times, stoploss.loss)