# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import linregress
from backtester.analysis.linregress import regression
import matplotlib.pyplot as plt

def test_regress():
    
    x = np.linspace(0, 1, 1000)
    noise = np.random.normal(0, .2, x.shape)
    y = -x + noise
    x1 = x[None, :]
    y1 = y[None, :]
    
    scipy_result = linregress(x, y)
    
    m, b, r = regression(x1, y1)
    print(scipy_result)
    
    assert np.isclose(m[0], scipy_result.slope)
    assert np.isclose(b[0], scipy_result.intercept)
    assert np.isclose(r[0], scipy_result.rvalue)
    
    
    
if __name__ == '__main__':
    test_regress()