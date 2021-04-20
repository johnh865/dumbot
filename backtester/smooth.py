# -*- coding: utf-8 -*-

import pdb 
from functools import cached_property
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from backtester.indicators import array_windows, TrailingBase
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar, root_scalar

import logging

logger = logging.getLogger(__name__)

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



class SmoothOptimize:
    """Optimize a filter for smoothing by inputted estimated signal amplitude
    and allowable signal acceleration. 

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    error_amp : float
        Estimated amplitude of signal. Amplitude below this is considered
        error, and the optimizer will attempt to filter them out.
    max_accel : float
        Estimated max accleration of the signal. The optimizer will attempt
        to smooth the 2nd derivative of the signal by limiting its
        acceleration. Specify the max expected acceleration (positive scalar)
        here. 
        
        Signal is expected to be in form:
            
            >>> x = func(t)
            >>> t = time array from [0, 1, 2, ...] or time change of 1.     
            
        Calculate `max_accel` accordingly. 
        
    signal_period : int, optional
        Overwrite `max_accel` by estimating the period of the signal in order 
        to filter high frequency noise. If `None`, use `max_accel`. 
        
        >>> max_accel = (pi / signal_period)**2 * error_amp
        
        
    polyorder : int, optional
        Filter order of Sav-Gol. See `scipy.signal`. The default is 5.
    max_window : int, optional
        Maximum allowed filter window size. See `scipy.signal`. The default is 500.
    """    
    def __init__(self,
            x: np.ndarray,
            error_amp: float, 
            max_accel : float,
            polyorder: int=5,
            max_window : int=500,
            signal_period : int=None
            ):
        self.max_accel = max_accel
        if signal_period is not None:
            self.max_accel = (np.pi / signal_period)**2 * error_amp
        self.unsmoothed = x
        self.error_amp = error_amp
        self.polyorder = polyorder
        self.max_window = max_window
        self._response_store = [None, None, None]
        self.dumb_smooth()
        
        

    def _post(self):
        x = self.unsmoothed
        self.position = self._response_store[0]
        self.acceleration = self._response_store[1]
        opt_window = self._response_store[2]
        self.speed = savgol_filter(x, opt_window, polyorder=3, deriv=1)
        self.opt_window = opt_window
        
        
    
    def dumb_smooth(self):
        """Do stupid optimization by looping through every window.
        Note that traditional algorithms have trouble due to existence
        of several local optima."""
        x = self.unsmoothed
        xlen = len(x)
        
        bound1 = self.polyorder + 1
        bound2 = min(xlen - 3, self.max_window)
        
        bound1 = self._get_window(bound1)
        bound2 = self._get_window(bound2)
        windows = np.arange(bound1, bound2, 2)
        
        residuals = []
        for ii in windows:
            resid = self.residual(ii)
            residuals.append(resid)
            
        imin = np.argmin(residuals)
        self.residual(windows[imin])
        self.result = (windows, residuals)
        self._post()


    def residual(self, window: int):
        """Calculate residual/objective for minimization."""
        x = self.unsmoothed
        error_amp = self.error_amp
        polyorder = self.polyorder
        
        window = self._get_window(window)
        xf = savgol_filter(x, window, polyorder=polyorder)
        d2xf = savgol_filter(x, window, polyorder=polyorder, deriv=2)
        
        # Positional error
        error = np.abs(x - xf)
        excess_error = np.mean(error) / error_amp
        
        # Acceleration error
        accel_error = np.abs(d2xf) - self.max_accel
        accel_error = np.maximum(accel_error, 0)
        excess_accel_error = np.mean(accel_error) / self.max_accel
        
        out = excess_error * 1 + excess_accel_error * 1
        
        self._response_store[0] = xf
        self._response_store[1] = d2xf
        self._response_store[2] = window        
        
        if logger.isEnabledFor(logging.DEBUG):
            string = (f'w={window:d}, e_amp={excess_error:.3f}, '
                      f'e_accel={excess_accel_error:.3f}, resid={out:.3f}')
            logger.debug(string)
        return out
    
    
    @staticmethod
    def _get_window(w: int):
        """Get window that can be accepted by savgol filter."""
        w = int(w)
        if w % 2 == 0:
            w += 1
        return w
    
    
    def filter(self, x: np.ndarray, **kwargs):
        """Filter data `x` using the optimized filter parameters."""
        opt_window = self._response_store[2]
        polyorder = self.polyorder 
        x0 = savgol_filter(x, 
                           window_length=opt_window, 
                           polyorder=polyorder,
                           **kwargs
                           )
        x1 = savgol_filter(x, 
                           window_length=opt_window,
                           polyorder=polyorder, 
                           deriv=1,
                           **kwargs
                           )
        x2 = savgol_filter(x,
                           window_length=opt_window,
                           polyorder=polyorder, 
                           deriv=2,
                           **kwargs)
        return x0, x1, x2


        
    def __smooth_opt_badly(self):
        """This function doesn't work so well as it tends to get the local,
        not global, optimum."""
        raise ValueError("Don't us this.")
        x = self.unsmoothed
        
        xlen = len(x)
        bounds = [5, xlen-3]
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
        
        self.result = res
        self._post()
        return 
    
class TrailingSavGol(TrailingBase):
    def __init__(self, series : pd.Series, window_size : int,
                 polyorder: int=5,):
        self.polyorder = polyorder
        self.window_size = np.int64(window_size)
        self.series = series   
       
         
    @cached_property
    @TrailingBase._append_nan_dec
    def position(self):
        closes = self._adj_close_intervals[:, -self.window_size:]
        output = savgol_filter(
            closes, 
            self.window_size,
            polyorder=self.polyorder,
            axis=1)
        return output[:, -1]
           
     
    @cached_property
    @TrailingBase._append_nan_dec
    def velocity(self):
        closes = self._adj_close_intervals[:, -self.window_size:]
        output = savgol_filter(
            closes, 
            self.window_size,
            polyorder=self.polyorder,
            deriv=1,
            axis=1)
        return output[:, -1]        

                
    @cached_property
    @TrailingBase._append_nan_dec
    def acceleration(self):
        closes = self._adj_close_intervals[:, -self.window_size:]
        output = savgol_filter(
            closes, 
            self.window_size,
            polyorder=self.polyorder,
            deriv=2,
            axis=1)
        return output[:, -1]   
    
    
def test1():
    ptnum = 1000
    period = 200
    x, y0, y0r = simple_noise_curve(ptnum, noise=.4, pt_period=period)
    
    
    # ys, window = smoother2(y0r, .4, percentile=60)
    sopt = SmoothOptimize(y0r, error_amp=1, max_accel=.003,
                          polyorder=5)
    sopt.dumb_smooth()
    ys = sopt.position
    d2ys = sopt.acceleration


    
    # ys, d2ys = smoother(y0r, period=20, error=.07)
    dy = np.gradient(y0)
    d2y = np.gradient(dy)
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x, y0, label='original')
    plt.plot(x, y0r, label='noise')
    plt.plot(x, ys,  label='smoothed')
    plt.grid()
    plt.legend()
    plt.title('Response vs Time')
    
    plt.subplot(3,1,2)
    plt.plot(x, d2y, label='original')
    plt.plot(x, d2ys, label='smoothed')
    plt.grid()
    plt.legend()
    plt.title('Response Acceleration vs Time')
    
    plt.subplot(3,1,3)
    plt.semilogy(sopt.result[0], sopt.result[1])
    plt.title('Residuals vs Filter Window Length')
    return sopt
    # pdb.set_trace()
    
if __name__ == '__main__':
    sopt = test1()