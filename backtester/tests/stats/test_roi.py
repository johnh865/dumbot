# -*- coding: utf-8 -*-
import pdb
import pytest
import pandas as pd
from os.path import join
from backtester.stockdata import YahooData2

from backtester.roi import ROI, ROIDaily
from backtester.definitions import PACKAGE_PATH

def get_dataframe():
    path = join(PACKAGE_PATH, 'tests', 'test_dataframe_SPY.csv')
    df = pd.read_csv(path)
    df = df.infer_objects()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df 


@pytest.fixture
def fixture_dataframe():
    return get_dataframe()    


def test1(fixture_dataframe):
    df = fixture_dataframe

    # y = YahooData2()
    # df = y.dataframes['SPY']

    r = ROI(df['Adj Close'], 100)

    r.index_interval_starts
    r.annualized
    r.annualized_adjusted



def test_daily(fixture_dataframe):
    df = fixture_dataframe
    r = ROIDaily(df['Adj Close'])
    r.annualized
    r.annualized_adjusted
    
    return


def test_FPC():
    y = YahooData2(symbols=['FPC'])
    series = y['FPC']['Adj Close']
    roi = ROI(series, 7)
    roi.times
    assert len(roi.times) < len(series)
    
    

    
if __name__ == '__main__':
    # test1(get_dataframe())
    # test_daily(get_dataframe())
    test_FPC()