"""There's about 252 days of trading per year."""


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from backtester.stockdata import YahooData, Indicators
from backtester.indicators import TrailingStats, trailing_mean
from backtester.definitions import DF_ADJ_CLOSE
yahoo = YahooData()
symbols = yahoo.get_symbol_names()
rs = np.random.default_rng(1)
rs.shuffle(symbols)

STOCKS = symbols[0:60]
STOCKS.append('SPY')
STOCKS.append('GOOG')
STOCKS.append('TSLA')



def post1(df:pd.DataFrame):
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, 252*5)
    stat1 = ts.exp_growth
    return stat1
    # stat2 = ts.exp_std_dev
    # return {'growth' : stat1, 'exp_std_dev' : stat2}

yahoo.symbols = STOCKS
indicator = Indicators(yahoo)
indicator.create(post1)


df = indicator.get_column_from_all('post1()')
# mean = df.mean().sort_values()
