# -*- coding: utf-8 -*-
# Num trading days per year = 252

import pandas as pd

from backtester.stockdata import YahooData, Indicators, MapData
from backtester.definitions import DF_ADJ_CLOSE
from backtester.indicators import TrailingStats


y_data = YahooData()
symbols = y_data.retrieve_symbol_names()


def roi(key):
    quarter = 252 // 4
    df = y_data.dataframes[key]
    s = df[DF_ADJ_CLOSE]
    t = TrailingStats(s, window_size=quarter, skip=quarter)
    metric = t.return_ratio
    
    return pd.Series(metric, index=t.times)
    



map_data = MapData(symbols, roi)
