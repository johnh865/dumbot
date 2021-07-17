# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from backtester.indicators import QuarterStats


from backtester.definitions import DF_ADJ_CLOSE, DF_VOLUME, DF_HIGH, DF_LOW
from datasets.symbols import ALL
from backtester.stockdata import YahooData




data = YahooData()
df = data.dataframes['SPY']
series = df[DF_ADJ_CLOSE]


q = QuarterStats(series)
start = q._index_start
end = q._index_end

slope = q.exp_growth
ratio = q.return_ratio