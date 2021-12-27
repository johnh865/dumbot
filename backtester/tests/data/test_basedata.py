# -*- coding: utf-8 -*-

from backtester.stockdata import YahooData, TableData
import pandas as pd
import pdb

def func1(table: TableData):
    return table.dataframe['Adj Close']


def test1():
    y = YahooData(symbols=('SPY', 'GOOG'))
    out = y.tmap(func1)
    series = out['SPY']
    
    assert type(series) == pd.Series
    assert series.name == 'Adj Close'
    
if __name__ == '__main__':
    test1()
    