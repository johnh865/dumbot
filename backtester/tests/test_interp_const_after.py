# -*- coding: utf-8 -*-
"""Speed test interp_const_after."""
import timeit
import pandas as pd
import numpy as np
from backtester.utils.misc import InterpConstAfter
from backtester.utils import floor_to_date
from backtester.utils.misc import (_interp_const_after_old1,
                                   _interp_const_after_old2)

def speedtest():
    x = np.arange(100)
    y = x**2
    x1 = np.linspace(5, 30, 100)
    
    def test1():
        return _interp_const_after_old1(x,y,x1)
    
    def test2():
        return _interp_const_after_old2(x, y, x1)
        
    def test3():
        return np.interp(x1, x, y)
        
    ic = InterpConstAfter(x, y)
    def test4():
        return ic.array(x1)
        
    def test5():
        return ic(x1)
    
    
    def test6():
        return InterpConstAfter(x, y).array(x1)
        
    test1()
    test2()
    test3()    
    test4()    
    test5()    
    test6()    
    t1 = timeit.timeit(test1, number=10000)
    t2 = timeit.timeit(test2, number=10000)
    t3 = timeit.timeit(test3, number=10000)
    t4 = timeit.timeit(test4, number=10000)
    t5 = timeit.timeit(test5, number=10000)
    t6 = timeit.timeit(test6, number=10000)
    print('old interp const1', t1)
    print('old interp const2', t2)
    print('numpy.interp', t3)
    print('new class', t4)
    print('new auto type', t5)
    print('new class re-called', t6)
    


def test_scalar():
    x = np.arange(100)
    y = x**2
    x1 = 23
    ic = InterpConstAfter(x, y)
    y1 = ic.scalar(x1)
    
    
    
    

def test_0():
    x = np.arange(10)
    y = x**2
    
    x0 = [-5.5, 0, 1, 1.5, 1.6, 2, 5.2, 6.7, 8, 20]
    x0 = np.array(x0)
    
    x0c = np.minimum(x.max(), x0)
    x0c = np.maximum(x.min(), x0c)
    y0 = np.floor(x0c)**2
    
    ic = InterpConstAfter(x, y)
    yt = ic.array(x0)
    assert np.all(y0 == yt)
    return



def test_1():
    x = np.arange(10.)
    y = x**2
    x0 = 4.4
    y0 = np.floor(x0)**2
    ic = InterpConstAfter(x, y)
    yt = ic.scalar(x0)
    assert yt == y0
    return


def test_2():
    x = np.arange(10)
    y = x**2
    x0 = [-4.2, 2.4, 6.2]
    x0 = np.array(x0)
    xbefore = -51
    
    y0 = np.floor(x0)**2
    y0[0] = xbefore
    
    ic = InterpConstAfter(x, y, xbefore)
    yt = ic.array(x0)
    assert np.all(y0 == yt)
    return


def test_3():
    
    x = np.array([np.datetime64('2016-01-01')])
    y = np.array([23])
    x0 = np.datetime64('2016-01-05')
    
    
    d = {'x':x, 'y':y}
    df = pd.DataFrame(d)
    
    ic = InterpConstAfter(df['x'], df['y'], before=0)
    yt = ic.scalar(x0)
    
    assert yt  == y[0]
    assert np.isscalar(yt)
    return


def test_4():
    
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
    
    ic = InterpConstAfter(df['x'], df['y'])
    yt = ic.scalar(x0)
    
    
    assert yt == 0
    return

    
    
    
    
    
    
if __name__ == '__main__':
    speedtest()
    test_scalar()
    test_0()
    test_1()
    test_2()
    test_3()
    test_4()
    
    
    