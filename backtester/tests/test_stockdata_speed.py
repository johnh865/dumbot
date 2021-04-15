# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import timeit

time1 = np.datetime64('2016-01-01')
time2 = np.datetime64('2021-01-01')
times = np.arange(time1, time2)
tlen = len(times)
d = {}
d['a'] = np.random.randn(tlen)
d['b'] = np.random.randn(tlen)
d['c'] = np.random.randn(tlen)
d['d'] = np.random.randn(tlen)


df = pd.DataFrame(d, index=times)
# df = df.sample(tlen)

class Prepper:
    def __init__(self, df: pd.DataFrame):
        df = df.sort_index()
        self.times = df.index
        self.df = df
        self.values = df.values
        self.names = df.columns.values.astype(str)
        self.df2 = df.copy()
        
        
    def get(self, date: np.datetime64):
        ii = np.searchsorted(self.times, date)
        return self.df.iloc[0 : ii]
    
    
    def get_np(self, date: np.datetime64):
        ii = np.searchsorted(self.times, date)
        v = self.values[0 : ii].T
        return dict(zip(self.names, v))


def get1(df: pd.DataFrame, date:np.datetime64):
    return df[df.index < date]



p = Prepper(df)

def test1():
    """Test prepper"""
    dates1 = times.copy()
    for date1 in dates1:
        p.get(date1)
        
        
def test2():
    dates1 = times.copy()
    for date1 in dates1:
        get1(df, date1)


def test3():
    """Test prepper, numpy"""
    dates1 = times.copy()
    for date1 in dates1:
        p.get_np(date1)
        


t1 = timeit.timeit(test1, number=2)
t2 = timeit.timeit(test2, number=2)
t3 = timeit.timeit(test3, number=2)
print(t1)
print(t2)
print(t3)

for date in times:
    assert np.all(p.get(date) == get1(df, date))
    