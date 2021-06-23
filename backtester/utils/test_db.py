# -*- coding: utf-8 -*-

TEST_DB_NAME = 'sqlite:///:memory:'
TEST_DF_NAME = 'test-db'

import pandas as pd
import numpy as np
from backtester.utils.db import SQLClient
import pdb

d = {}
time1 = np.datetime64('2004-01-01')
time2 = np.datetime64('2008-01-01')
times = np.arange(time1, time2)
d['x'] = np.linspace(0, 1, len(times))
d['y'] = d['x']**2
d['z'] = d['x']**3




def test_save_and_read():
    df = pd.DataFrame(d, index=times)    
    client = SQLClient(TEST_DB_NAME)
    client.save_dataframe(df, TEST_DF_NAME)
    df2 = client.read_dataframe(TEST_DF_NAME)
    
    assert np.all(df2.values == df.values)
    
    
    
def test_append1():
    df = pd.DataFrame(d, index=times)    
    client = SQLClient(TEST_DB_NAME)
    client.save_dataframe(df, TEST_DF_NAME)
    df2 = client.append_dataframe(df, TEST_DF_NAME)
    df3 = client.read_dataframe(TEST_DF_NAME)
    assert np.all(df2.values == df.values)
    assert np.all(df3.values == df.values)
    

def test_append2():
    df = pd.DataFrame(d, index=times)    
    df1 = df.iloc[0:700]
    df2 = df.iloc[500:]
    
    client = SQLClient(TEST_DB_NAME)
    client.save_dataframe(df1, TEST_DF_NAME)
    
    df3 = client.append_dataframe(df2, TEST_DF_NAME)
    assert np.all(df3.values == df.values)
    df3 = client.read_dataframe(TEST_DF_NAME)
    assert np.all(df3.values == df.values)
    
    
if __name__ == '__main__':
    # test_append1()
    test_append2()