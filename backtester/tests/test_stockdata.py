# -*- coding: utf-8 -*-
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtester.stockdata import YahooData, Indicators, TableData
from backtester.definitions import DF_ADJ_CLOSE
from datetime import datetime

def test_yahoo():
    
    y = YahooData(['DIS', 'GOOG'])
    date = datetime(2016, 5, 23)
    date = np.datetime64(date)
    out = y.filter_dates(end=date)
    
    # out  = y.filter_before(date)
    # y.get_symbols_before(date)
    df3 = out.dataframes['GOOG']
    assert df3.index[-1] <= date
    return


def trailing_avg(period, df: pd.DataFrame):
    close = df[DF_ADJ_CLOSE]
    return close.rolling(period).mean()
    

def test_indicator():
    y = YahooData(['DIS', 'GOOG'])
    indicators = Indicators(y)
    indicators.create(trailing_avg, 5, name='trailing')
    df1 = indicators.dataframes['DIS']
    df2 = indicators.dataframes['GOOG']
    assert 'trailing' in df1.columns
    assert 'trailing' in df2.columns
    
    
def test_get_trade_dates():
    
    y = YahooData(['DIS', 'GOOG'])
    dates = y.get_trade_dates()
    
    
    delta = dates[1:] - dates[0:-1]
    delta = delta.astype('timedelta64[D]')
    assert np.max(delta) < 10
    plt.plot(dates, dates, '.')
    return


def test_table_data():
    y = YahooData(['DIS'])
    table: TableData = y.tables['DIS']
    
    date = np.datetime64('2016-01-01')
    ii = table.date_index(date)
    
    a1 = table.array_up_to(date)
    a2 = table.values[0 : ii + 1]
    a3 = table.dataframe_up_to(date).values
    assert np.all(a1 == a2)
    assert np.all(a1 == a3)


    return




if __name__ == '__main__':
    test_table_data()
    test_yahoo()
    test_indicator()
    test_get_trade_dates()
    test_table_data()