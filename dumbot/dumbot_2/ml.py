# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester.indicators import TrailingStats
from backtester.stockdata import YahooData
from backtester.definitions import DF_ADJ_CLOSE
from backtester.analysis import BuySell, avg_future_growth

from backtester import utils

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor


class CreateData:
    def __init__(self, symbol: str):
        yahoo_data = YahooData([symbol])
        self.df = yahoo_data.retrieve_symbol(symbol)
        if len(self.df) == 0:
            raise ValueError(f'Symbol {symbol} data not available') 
            
            
    def features(self, window: int):
        attrs = ['exp_growth', 'exp_std_dev']
        df = self.df
        series1 = df[DF_ADJ_CLOSE]
        
        ts1 = TrailingStats(series1, window)

        new = {}
        for attr in attrs:
            feature = getattr(ts1, attr)
            name = f'Close({attr}, {window})'
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




def test2():
    """Test regression."""    
    

    yahoo = YahooData()
    symbols = yahoo.get_symbol_names()
    np.random.seed(1)
    np.random.shuffle(symbols)
    symbols = symbols[0 : 5]
    
    
    windows1 = [40, 200]
    creator = CreateData('MSFT')
    y = creator.targets(40)



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



yahoo = YahooData()
symbols = yahoo.get_symbol_names()
np.random.seed(1)
np.random.shuffle(symbols)
symbols = symbols[0 : 5]

for symbol in symbols:
    df = yahoo[symbol][DF_ADJ_CLOSE]
