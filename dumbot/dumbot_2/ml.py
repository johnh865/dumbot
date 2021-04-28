# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester.indicators import TrailingStats, FutureStats
from backtester.stockdata import YahooData
from backtester.definitions import DF_ADJ_CLOSE
from backtester.analysis import BuySell, avg_future_growth

from backtester import utils
from backtester.exceptions import NotEnoughDataError
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor


class CreateData:
    def __init__(self, feature_windows: list[int], target_window: int):
        self.feature_windows = feature_windows
        self.target_window = target_window
        
    
    
    def build(self, symbols: list[str]):
        
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        
        for symbol in symbols:
            try:
                X_train1, y_train1, X_test1, y_test1 = self.build_one(symbol)
                
                X_train_list.append(X_train1)
                y_train_list.append(y_train1)
                X_test_list.append(X_test1)
                y_test_list.append(y_test1)
            except NotEnoughDataError:
                pass
            
            
        X_train = np.row_stack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.row_stack(X_test_list)
        y_test= np.concatenate(y_test_list)
        
        # Find and get ri of NAN values???
        # How'd they slip in anyways???
        train_mask = ~np.max(np.isnan(X_train), axis=1)
        train_mask = train_mask & ~np.isnan(y_train)
        
        test_mask = ~np.max(np.isnan(X_test), axis=1)
        test_mask = test_mask & ~np.isnan(y_test)
        
        return (
            X_train[train_mask],
            y_train[train_mask], 
            X_test[test_mask],
            y_test[test_mask],
            )
        
    
    def build_one(self, symbol:str):
        
        # feature_windows = [200, 400]
        # target_window = 400
        feature_windows = self.feature_windows
        target_window = self.target_window
        split_date = np.datetime64('2016-01-01')


        yahoo = YahooData()
        df = yahoo[symbol]
        targets = self._targets(df, target_window)
        flist = []
        for window in feature_windows:
            features = self._features(df, window)
            flist.append(features)
        
        X1, y1 = combine_datas(flist, targets)
            
        split_index = np.searchsorted(X1.index, split_date)
        X_train1 = X1[0 : split_index]
        y_train1 = y1[0 : split_index]
        
        X_test1 = X1[split_index :]
        y_test1 = y1[split_index :]
        
        return X_train1, y_train1, X_test1, y_test1
                        

            
    def _features(self, df, window: int):
        attrs = ['exp_growth', 'exp_std_dev']
        series1 = df[DF_ADJ_CLOSE]
        
        ts1 = TrailingStats(series1, window)

        new = {}
        for attr in attrs:
            feature = getattr(ts1, attr)
            
            # Standardize exp_std_dev
            if 'exp_std_dev' in attr:
                mean = np.nanmean(feature)
                std = np.nanstd(feature)
                feature = (feature - mean) / std
            
            name = f'Close({attr}, {window})'
            new[name] = feature
        
        df_new = pd.DataFrame(new, index=df.index)
        df_new = df_new.iloc[window :]
        return df_new
    
    
    def _targets(self, df, window: int):
        attrs = ['exp_growth']
        series1 = df[DF_ADJ_CLOSE]
        
        ts1 = FutureStats(series1, window)

        new = {}
        for attr in attrs:
            feature = getattr(ts1, attr)
            name = f'FutureGrowth({attr}, {window})'
            new[name] = feature
        
        df_new = pd.DataFrame(new, index=df.index)
        df_new = df_new.iloc[0 : -window]
        return df_new
        
    
    
def combine_datas(features_list: list[pd.DataFrame], target: pd.DataFrame):
    target_name = target.columns[0]
    features = pd.concat(features_list, axis=1, join='inner',)
    feature_names = features.columns
    # new = pd.merge(target, features, how='inner')
    # new = pd.concat([features, target], join='outer')
    new = target.join(features, how='inner')
    return new[feature_names], new[target_name]

windows1 = np.arange(300, 5, -10)
windows2 = np.arange(100, 5, -10)

yahoo = YahooData()
symbols = yahoo.get_symbol_names()
rs = np.random.RandomState(0)
rs.shuffle(symbols)
symbols1 = symbols[0:2]
symbols1 = ['SPY', 'GOOG']

scores = []
for window2 in windows2:        
    for window1 in windows1:
        c = CreateData(feature_windows=[window1], target_window=window2)

        
        X_train, y_train, X_test, y_test = c.build(symbols1)
        
        
        reg = RandomForestRegressor()
        
        reg.fit(X_train, y_train)
        score0 = reg.score(X_train, y_train)
        score = reg.score(X_test, y_test)
        print(window1, window2, score, score0)
        
        
        scores.append(score)
        y_pred = reg.predict(X_test)
        # plt.plot(y_pred, y_test, '.', alpha=.2)
        
        
        
        
        