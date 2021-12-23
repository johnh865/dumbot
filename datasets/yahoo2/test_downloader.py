# -*- coding: utf-8 -*-

import pytest
import datetime
import numpy as np

from datasets.yahoo2.downloader import (
    YahooClient, download_df_dict, download_df_yr_dict
    )
from datasets.symbols import ALL as ALL_SYMBOLS




        
        
def test_download_df_yr_dict():
    return download_df_yr_dict(['SPY', 'GOOG', 'MSFT'], 1995)



def test_download_yesterday():
    # yesterday = today - datetime.timedelta.days(1)
    date1 = '2021-10-20'
    date2 = '2021-10-21'
    
    d = download_df_dict(ALL_SYMBOLS, start_date=date1, end_date=date2)
    return d


def test_download_all():
    return download_df_dict(ALL_SYMBOLS,)
    


def downloader_tester(date1, date2, date3):
    
    path = 'test_yahoo_downloader'
    yd = YahooClient(path, symbols=('SPY', 'GOOG', 'MSFT'))
    yd.init_data(
        start_date = date1,
        end_date = date2)
    
    df = yd.read('SPY')
    
    yd.update(date3)
    df2 = yd.read('SPY')
    
    
    yd2 = YahooClient(path, symbols=('SPY', 'GOOG', 'MSFT'))
    yd2.init_data(start_date = date1, end_date = date3)
    df3 = yd2.read('SPY')
    
    
    # Check that update and init construct same dataframe.
    assert np.all(np.isclose(df2.values, df3.values))
    assert np.all(df2.index == df3.index)
    
    # Check time index is in the specified range
    assert np.all((df.index >= date1) & (df.index <= date2))
    yd2.client.delete()
    

    
def test_downloader1():
    date1 = '2017-01-02'
    date2 = '2017-02-01'
    date3 = '2017-03-15'
    
    downloader_tester(date1, date2, date3)
    
    
def test_downloader2():
    date1 = '2018-01-02'
    date2 = '2018-01-05'
    date3 = '2018-01-06'
    
    downloader_tester(date1, date2, date3)
    
    
if __name__ == '__main__':
    test_downloader1()
    test_downloader2()