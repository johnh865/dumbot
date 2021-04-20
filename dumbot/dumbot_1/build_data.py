# -*- coding: utf-8 -*-
"""Attempt at constructing classification of future growth/drop. Failure."""

import pdb

import numpy as np
import pandas as pd
import backtester

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from backtester.indicators import TrailingStats
from backtester.analysis import BuySell, avg_future_growth
from backtester.stockdata import YahooData, read_yahoo_symbol_names
from backtester.definitions import DF_ADJ_CLOSE, DF_VOLUME
from backtester import utils

from datasets.symbols import ALL


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt

class CreateData:
    def __init__(self, symbol: str):
        yahoo_data = YahooData([symbol])
        self.df = yahoo_data.retrieve_symbol(symbol)
        if len(self.df) == 0:
            raise ValueError(f'Symbol {symbol} data not available') 
            
            
    def features(self, window: int):
        # attrs = ['exp_growth', 'exp_accel', 'exp_reg_diff']
        attrs = ['exp_growth', 'exp_accel']
        # attrs = ['exp_accel', ]
        df = self.df
        series1 = df[DF_ADJ_CLOSE]
        series2 = df[DF_VOLUME]
        
        ts1 = TrailingStats(series1, window)
        ts2 = TrailingStats(series2, window)

        new = {}
        for attr in attrs:
            feature = getattr(ts1, attr)
            name = f'Close({attr}, {window})'
            new[name] = feature

        feature = ts2.lin_reg_value
        name = f'Volume({window})'
        new[name] = feature
        
        df_new = pd.DataFrame(new, index=df.index)
        df_new = df_new.iloc[window :]
        return df_new

        
    def targets(self, window: int):
        
        # Get future growths
        df = self.df
        series1 = df[DF_ADJ_CLOSE]
        times = utils.dates2days(series1.index.values)
        dates = series1.index[0 : -window]

        _, growths = avg_future_growth(times,
                                       series1.values, 
                                       window=window)  
        d = {}
        d['avg_future_growths'] = growths
        df = pd.DataFrame(d, index=dates)
        return df
    
def combine_datas(features_list: list[pd.DataFrame], target: pd.DataFrame):
    target_name = target.columns[0]
    features = pd.concat(features_list, axis=1, join='inner',)
    feature_names = features.columns
    # new = pd.merge(target, features, how='inner')
    # new = pd.concat([features, target], join='outer')
    new = target.join(features, how='inner')
    return new[feature_names], new[target_name]

    
def create_data(symbol: str, window: int, future_window: int):
    attrs = ['exp_growth', 'exp_accel', 'exp_reg_diff']
    yahoo_data = YahooData([symbol])
    df = yahoo_data.retrieve_symbol(symbol)
    if len(df) == 0:
        raise ValueError(f'Symbol {symbol} data not available')
    else:
        series1 = df[DF_ADJ_CLOSE]
        series2 = df[DF_VOLUME]
        
        ts1 = TrailingStats(series1, window)
        ts2 = TrailingStats(series2, window)
        
        new = {}
        feature_names = []
        for attr in attrs:
            feature = getattr(ts1, attr)
            name = f'Close({attr}, {window})'
            new[name] = feature[0 : -future_window]
            feature_names.append(name)

        # feature = ts2.lin_reg_value
        # name = f'Volume({window})'
        # new[name] = feature[0 : -future_window]            
        # feature_names.append(name)
        
        # Get future growths
        times = utils.dates2days(series1.index.values)
        dates = series1.index[0 : -future_window]
    
        _, growths = avg_future_growth(times,
                                       series1.values, 
                                       window=future_window)        
        

        new['avg_future_growth'] = growths
        new['date'] = dates
        # new[DF_ADJ_CLOSE] = series.values[0 : -future_growth_window]
        df_new = pd.DataFrame(new)
        df_new = df_new.iloc[window :]
        # Set date to indx
        df_new = df_new.set_index('date')
        return df_new[feature_names], df_new['avg_future_growth']



def test1():    
    windows1 = [10, 20, 50, 100, 200]
    windows2 = [10, 20, 50,]
    wmg1, wmg2 = np.meshgrid(windows1, windows2)
    
    wr1 = wmg1.ravel()
    wr2 = wmg2.ravel()
    reg = LinearRegression()
    
    scores = []
    for window1, window2 in zip(wr1, wr2):
        print(window1, window2)
        x, y = create_data('GOOG', window1, window2)
        
        
        scaler_x = preprocessing.StandardScaler().fit(x.values)
        x1 = scaler_x.transform(x)
        reg.fit(x1, y)
        score = reg.score(x, y)
        print(score)
        scores.append(score)
        
        # 
        # plt.figure()
        # for values, name in zip(x1.T, x):
        #     plt.plot(values, y, '.', alpha=.4, label=name)
        # plt.legend()
    
    scores = np.array(scores).reshape(wmg1.shape)


def test2():
    """Test regression."""    
    windows1 = [5, 20, 200]
    creator = CreateData('MSFT')
    
    y = creator.targets(50)
    xlist = []
    for window in windows1:
        print(f'building window {window}')
        x = creator.features(window)
        xlist.append(x)
    x1, y1 = combine_datas(xlist, y)
    yc1 = y1 >= 0
    
    split_date = np.datetime64('2016-01-01')
    split_index = np.searchsorted(x1.index, split_date)
    x_train = x1[0 : split_index]
    y_train = y1[0 : split_index]
    x_test = x1[split_index :]
    y_test = y1[split_index :]
    
    
    # clf = RandomForestClassifier()
    # clf.fit(x_train, y_train)
    # score =clf.score(x_test, y_test)
    # yc_predict = clf.predict(x_test)

    depths = [5]
    for depth in depths:
        reg = RandomForestRegressor(max_depth=depth)
        reg.fit(x_train, y_train)
        score = reg.score(x_test, y_test)
        y_predict = reg.predict(x_test)
        print(depth, score)
    plt.plot(y_predict, y_test, '.', alpha=.2)
        
    
def test3():
    """test.
    FAILURE! Neural net sucks at regression!"""
    windows1 = [10, 50, 200]
    creator = CreateData('MSFT')
    y = creator.targets(50)    
    
    
    xlist = []
    for window in windows1:
        print(f'building window {window}')
        x = creator.features(window)
        xlist.append(x)
    x1, y1 = combine_datas(xlist, y)    

    split_date = np.datetime64('2016-01-01')
    split_index = np.searchsorted(x1.index, split_date)

    # Scale
    scaler_x = preprocessing.StandardScaler().fit(x1)    
    x1 = scaler_x.transform(x1)    
    
    
    

    x_train = x1[0 : split_index]
    y_train = y1[0 : split_index]
    x_test = x1[split_index :]
    y_test = y1[split_index :]    
    
    regr = MLPRegressor([100]*100, 
                        random_state=1,
                        verbose=True,
                        max_iter=500)
    regr = regr.fit(x_train, y_train)
    
    score = regr.score(x_train, y_train)
    print('train score', score)
    
    score = regr.score(x_test, y_test)
    print('test score', score)
    pdb.set_trace()
    
    
    
def test4():
    """Test classification into UP or DOWN.
    FAILURE! Not able to classify whatsoever downward movements."""
    windows1 = [20, 40, 70, 100, 200, ]
    creator = CreateData('MSFT')
    y = creator.targets(20) 
    
    xlist = []
    for window in windows1:
        print(f'building window {window}')
        x = creator.features(window)
        xlist.append(x)
    x1, y1 = combine_datas(xlist, y) 
    
    split_date = np.datetime64('2016-01-01')
    split_index = np.searchsorted(x1.index, split_date)
    
    scaler_x = preprocessing.StandardScaler().fit(x1)    
    x1 = scaler_x.transform(x1)  
    
    # Turn into classification problem.
    y1 = y1 > 0
    
    # Weight down-turns more heavily
    weights = np.ones(len(y1))
    weights[y1 == 0] = 10
    
    # Split data
         
    x_train = x1[0 : split_index]
    y_train = y1[0 : split_index]
    w_train = weights[0 : split_index]
    x_test = x1[split_index :]
    y_test = y1[split_index :]    
        
    
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    
    score = clf.score(x_train, y_train, w_train)
    print('train score', score)
    
    score = clf.score(x_test, y_test)
    y_predict = clf.predict(x_test)
    print('test score', score)    
    
    # See how many negatives it got wrong
    ii = y_test == 0
    score2 = np.sum(y_predict[ii] == y_test[ii]) / len(y_predict[ii])
    print('test score predicting down', score2)
    pdb.set_trace()
    
    
if __name__ == '__main__':
    test4()
    
    