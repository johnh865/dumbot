# -*- coding: utf-8 -*-
import numpy as np
from backtester.stockdata import LazyMap
from math import sqrt
import pdb
import copy
# Keep track of every time function `func` is called. 
eval_record = []

def func(x):
    out = x ** 2
    eval_record.append(out)
    return out


def func_sqrt(x):
    return sqrt(x)

def check_list(a, b):
    a = np.array(a)
    b = np.array(b)
    assert np.all(a == b)

keys = np.arange(10)
values = keys ** 2

lm = LazyMap(keys, func)

# %% Tests

def test_keys():
    check_list(keys, list(lm.keys()))
    
    
def test_values():
    check_list(values, list(lm.values()))


def test_getitem():
    for ii, key in enumerate(keys):
        value = lm[key]
        assert value == values[ii]
        
        
    for key in keys:
        value = lm[key]
        
    # Make sure func is only called once per call. 
    assert len(eval_record) == len(keys)
        

def test_len():
    assert len(lm) == len(keys)        
    
    
def test_apply():
    lm1 = LazyMap(keys, func)
    
    # Test lazy evalution of 0 and 1 keys.
    lm1[0]
    lm1[1]
    
    func_dict = copy.deepcopy(lm1._func_dict)
    assert len(func_dict[0]) == 0
    assert len(func_dict[1]) == 0
    for ii in range(2, 10):
        assert len(func_dict[ii]) == 1
    
    # Apply another function
    lm1.apply(func_sqrt)
    
    # Test lazy evalution of 2 and 3 keys.
    lm1[2]
    lm1[3]    
    func_dict = copy.deepcopy(lm1._func_dict)
    
    assert len(func_dict[0]) == 1
    assert len(func_dict[1]) == 1
    assert len(func_dict[2]) == 0
    assert len(func_dict[3]) == 0
    for ii in range(4, 10):
        assert len(func_dict[ii]) == 2
    
       
    check_list(keys, list(lm1.values()))
    func_dict = copy.deepcopy(lm1._func_dict)
    for ii in range(10):
        assert len(func_dict[ii]) == 0
    
    
def test_add():
    keys = range(12)
    lm1 = LazyMap(keys, func)
    lm1[0]
    lm1[2]
    lm1.apply(func)
    lm1[3]
    lm1.apply(func)
    lm1[4]
    lm1.add(12)
    
    keys2 = np.arange(13)
    out2 = keys2**8
    out1 = np.array(list(lm1.values()))    
    check_list(out1, out2)
        
    
if __name__ == '__main__':
    test_keys()
    test_values()
    test_getitem()
    test_len()
    test_apply()
    test_add()








