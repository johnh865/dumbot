# -*- coding: utf-8 -*-
from itertools import product


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from datasets.fakedata.definitions import TRADE_DATES_PATH
from backtester.stockdata import BaseData
from backtester import utils
trade_dates = np.genfromtxt(TRADE_DATES_PATH, dtype=np.datetime64)

x = utils.dates2days(trade_dates)
xlen = len(x)
data_dict = {}

# %% Create SIN WAVES


def offset(d: dict, offset: float):
    new = {}
    for key, value in d.items():
        new[key] = value + offset
    return new

class FakeBuilder:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.dict = self.permute()
        
        
        
    def permute(self):
        sins = self.sin()
        noises = self.noise()
        walks = self.random_walk()
        
        d = {}
        d.update(sins)
        d.update(noises)
        d.update(walks)

        for key_s, key_n in product(sins, noises):
            key = key_s + key_n
            d[key] = sins[key_s] + noises[key_n]
            
        for key_s, key_n in product(sins, walks):
            key = key_s + key_n
            d[key] = sins[key_s] + walks[key_n]
            
            
        for key_s, key_n, key_r in product(sins, noises, walks):
            key = key_s + key_n + key_r
            d[key] = sins[key_s] + noises[key_n] + walks[key_r]
                        
        d = offset(d, 5.0)
        return d    
        
        
    def sin(self):
        data_dict = {}
        periods = [10, 20, 50, 500]
        for period in periods:
            name = f'Sin({period})'
            data_dict[name] = np.sin(2 * np.pi * x / period)
        return data_dict
            
            
    def noise(self):
        data_dict = {}
        noises = [0.1, 0.5, 1.0, ]
        for noise in noises:    
            name = f'Noise({noise})'
            data_dict[name] = self.rng.normal(scale=noise, size=xlen)
        return data_dict

            
    def random_walk(self):
        data_dict = {}
        noises = [0.01, 0.05, .1,]
        for noise in noises:    
            name = f'RandWalk({noise})'
            r = self.rng.normal(scale=noise, size=xlen)
            data_dict[name] = np.cumsum(r)
        return data_dict



f = FakeBuilder(0)
data_dict = f.dict




class FakeData(BaseData):
    def __init__(self, seed=0, include=()):
        self.fakebuilder = FakeBuilder(seed)
        self._dict = self.fakebuilder.dict
        
        _all_symbols = list(self._dict.keys())
        if not include:
            self._symbols = _all_symbols
        else:
            self._symbols = include
            
        
        
        
    def retrieve_symbol_names(self):
        return self._symbols


    def retrieve(self, symbol: str):
        times = trade_dates
        values = self._dict[symbol]    
        df = pd.DataFrame(data={'Close' : values}, index=times)
        return df
    
    
    def get_trade_dates(self):
        return trade_dates
