# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtester.utils import floor_to_date, interp_const_after
import datetime

def test_floor_to_date1():
    d1 = np.datetime64('1970-01-11T00:00:12')
    d2 = floor_to_date(d1)    
    
    r1 =  np.datetime64('1970-01-11T00:00:00')
    assert r1 == d2
    return


def test_floor_to_date2():
    d1 = datetime.datetime(2014, 5, 3, minute=12)
    r1 = datetime.datetime(2014, 5, 3, )
    d2 = floor_to_date(d1)    
    assert r1 == d2
    return

def test_floor_to_date3():
    d1 = np.datetime64('1970-01-11T00:00:00')
    d2 = floor_to_date(d1)    
    
    r1 =  np.datetime64('1970-01-11T00:00:00')
    assert r1 == d2
    return


def test_values_between():
    x = np.arange(10)
    y = x**2
    
    x0 = [-5.5, 0, 1, 1.5, 1.6, 2, 5.2, 6.7, 8, 20]
    x0 = np.array(x0)
    
    x0c = np.minimum(x.max(), x0)
    x0c = np.maximum(x.min(), x0c)
    y0 = np.floor(x0c)**2
    
    yt = interp_const_after(x, y, x0)
    assert np.all(y0 == yt)
    return



def test_values_between2():
    x = np.arange(10)
    y = x**2
    x0 = 4.4
    y0 = np.floor(x0)**2
    yt = interp_const_after(x, y, x0)
    assert yt == y0
    return


def test_values_between3():
    x = np.arange(10)
    y = x**2
    x0 = [-4.2, 2.4, 6.2]
    x0 = np.array(x0)
    xbefore = -51
    
    y0 = np.floor(x0)**2
    y0[0] = xbefore
    yt = interp_const_after(x, y, x0, xbefore)
    assert np.all(y0 == yt)
    return


def test_values_between4():
    
    x = np.array([np.datetime64('2016-01-01')])
    y = np.array([23])
    x0 = np.datetime64('2016-01-05')
    
    
    d = {'x':x, 'y':y}
    df = pd.DataFrame(d)
    yt = interp_const_after(df['x'], df['y'], x0, before=0)
    
    assert yt  == y[0]
    assert np.isscalar(yt)
    return


def test_values_between5():
    
    x = np.array([
        np.datetime64('2016-01-01'),
        np.datetime64('2016-03-01'),
        np.datetime64('2016-04-01'),
        ])    
    y = np.array([5, 10, 0])
    x0 = np.datetime64('2016-06-23')
    x0 = np.array(x0)
    d = {'x':x, 'y':y}
    df = pd.DataFrame(d)
    
    yt = interp_const_after(df['x'], df['y'], x0)
    assert yt == 0
    return

    
if __name__ == '__main__':
    # test_floor_to_date1()
    # test_floor_to_date2()
    # test_floor_to_date3()
    test_values_between()
    test_values_between2()
    test_values_between3()
    test_values_between4()
    test_values_between5()
    
    
    
    
    
    