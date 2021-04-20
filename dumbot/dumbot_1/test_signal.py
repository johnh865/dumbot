# -*- coding: utf-8 -*-
"""Test creating smoothing system..."""
import pdb 
import numpy as np
import matplotlib.pyplot as plt

import scipy
from backtester.indicators import array_windows
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar, root_scalar

def random_walk_curve(ptnum, noise):
    
    x0 = np.linspace(0, 10, ptnum)
    y0 = np.sin(2*np.pi*x0) + x0/2 + 10
        
    rstate = np.random.default_rng(seed=0)
    r_amp = noise
    r = r_amp * rstate.normal(size=ptnum)
    rwalk = np.cumsum(r)

    y0r = y0 + rwalk
    return x0, y0, y0r



def simple_noise_curve(ptnum, noise, pt_period):
    x0 = np.arange(0, ptnum)
    
    
    y0 = np.sin(2*np.pi*x0 / pt_period) + 10
    
    r_amp = noise
    rstate = np.random.default_rng(seed=0)
    r = r_amp * rstate.normal(size=ptnum)
    y0r = y0 + r
    return x0, y0, y0r


def smoother(x, period: int, error: float, growth=1.05):    
    
    max_accel = (2 * np.pi / period)**2 * error
    print('max accel', max_accel)
    xlen = len(x)
    
    window = int(period / 3)
    window = max(5, window)
    window = min(xlen / 2, window)    
    
    if window % 2 == 0:
        window += 1
    window = int(window)

    
    for ii in range(100):
        if window >= xlen:
            window = xlen
            if window % 2 == 0:
                window = window - 1 
            accel = savgol_filter(x, window, polyorder=3, deriv=2)
            break
            
        accel = savgol_filter(x, window, polyorder=3, deriv=2)
        print(accel.max(), window)
        if accel.max() < max_accel:
            break
        window = int(np.ceil(window * growth))
        if window % 2 == 0 :
            window += 1
    
    ynew = savgol_filter(x, window, polyorder=3)
    return ynew, accel


def smoother2(x, perror, percentile=75):
    response_store = [None, None]
    
    def residual(window):
        window = int(window)
        if window % 2 == 0:
            window = window + 1
        response = savgol_filter(x, window, polyorder=3)
        error = np.abs(x - response)
        
        max_error = np.percentile(error, percentile)
        
        response_store[0] = response
        response_store[1] = window
        out = perror - max_error
        print(window, out)
        
        return out
        
    xlen = len(x)
    bounds = [5, xlen-3]
    res = root_scalar(
        residual,
        bracket=bounds,
        # bracket=bracket,
        xtol=1.5,
        # method='brent'
        # method='bounded'
        method='brentq'
        )
    return response_store
    


def smoother3(x: np.ndarray,
              error_amp: float, 
              noise_period: int=2,):
    deriv = 2
    def get_window(w: int):
        w = int(w)
        if w % 2 == 0:
            w += 1
        return w
    
    max_accel = (np.pi / noise_period)**deriv * error_amp
    response_store = [None, None]
    
    def residual(window: int):
        window = get_window(window)
        xf = savgol_filter(x, window, polyorder=3)
        d2xf = savgol_filter(x, window, polyorder=3, deriv=deriv)
        
        # Positional error
        error = np.abs(x - xf)
        excess_error = np.mean(error) / error_amp
        
        # Acceleration error
        accel_error = np.abs(d2xf)
        excess_accel_error = np.mean(accel_error) / max_accel
        
        out = excess_error * 1 + excess_accel_error * 1
        
        response_store[0] = xf
        response_store[1] = window        
        
        string = (f'w={window:d}, e_amp={excess_error:.3f}, '
                  f'e_accel={excess_accel_error:.3f}, resid={out:.3f}')
        print(string)
        return out
    
    xlen = len(x)
    bounds = [5, xlen/3]
    res = minimize_scalar(
        residual,
        bounds=bounds,
        # bracket=bracket,
        # tol=1.5,
        method='bounded',
        options = {'xatol' : 1.5},
        # method='golden'
        # method='brentq'
        )
    return response_store
        
        

def test1():
    ptnum = 1000
    period = 100
    x, y0, y0r = simple_noise_curve(ptnum, noise=.4, pt_period=period)
    
    
    # ys, window = smoother2(y0r, .4, percentile=60)
    ys, window = smoother3(y0r, error_amp=0.4, noise_period=10, )
    d2ys = savgol_filter(y0r, window, polyorder=3, deriv=2)
    
    # ys, d2ys = smoother(y0r, period=20, error=.07)
    dy = np.gradient(y0)
    d2y = np.gradient(dy)
    
    plt.subplot(2,1,1)
    plt.plot(x, y0, label='original')
    plt.plot(x, y0r, label='noise')
    plt.plot(x, ys, label='smoothed')
    plt.grid()
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(x, d2y, label='original')
    plt.plot(x, d2ys, label='smoothed')
    plt.grid()
    plt.legend()
    
    
if __name__ == '__main__':
    test1()