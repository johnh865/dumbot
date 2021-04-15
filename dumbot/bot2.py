# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from backtester import Strategy, Backtest
from backtester.indicators import TrailingStats
from backtester.definitions import DF_ADJ_CLOSE
from backtester.stockdata import YahooData


from dumbot.build_data2 import load_db, load_symbol_names


from dumbot.build_data2 import create_data

import datetime


from sklearn import linear_model, svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, TimeSeriesSplit



# %% Load data
stocks = ['SPY', 
          'GOOG',
          'MSFT',
          'AAPL',
          # 'TSLA',
          'AIG',
          'ALK',
          'GRA',
          'HAL',
          'CR',
          ]


data_dict = create_data(stocks)

# %%
df = pd.concat(data_dict.values())

def process(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Extract features and targets for ML."""
        
    dates = df['date']
    date_min = dates.min()
    time_delta = dates - dates.min()
    
    col_names = df.keys()
    feature_names = [c for c in col_names if c.endswith(')')]
    features = df[feature_names]
    targets = df['avg_future_growth']
    return features, targets



features, targets = process(df)


# %% Machine learning
reg = RandomForestRegressor()


def train(features, targets, reg):
    
    scaler_x = preprocessing.StandardScaler().fit(features.values)
    # scaler_y = preprocessing.StandardScaler().fit(targets.values[:, None])
    
    features1 = scaler_x.transform(features)
    # targets1 = scaler_y.transform(targets.values[:, None]).ravel()
        
    # x_train, x_test, y_train, y_test = train_test_split(features1, 
    #                                                     targets, 
    #                                                     test_size=0.4,
    #                                                     random_state=0)

    
    x_train = features
    y_train = targets
    reg.fit(x_train, y_train)    
    score = reg.score(x_train, y_train)
    print('SCore=', score)
    return reg



def load_ml_data(dummy):
    return features.values, targets.values


def get_indicators(df: pd.DataFrame):
    
# %% Strategy & Performance

class Strat1(Strategy):
    
    def init(self):
        self.reg = RandomForestRegressor()
        
        self.ml_data = self.indicator(load_data)
        self.ii = 0
        return
    
    
    def next(self):
        # Train every XX days 
        features, targets = self.datas(stocks[0])
        
        if (self.ii + 1) % 100 == 0:
            self.reg.fit(features, targets)
            
        future_growth = self.reg.predict(features[-1:])
        return
    
    
if __name__ == '__main__':
    y = YahooData(symbols=stocks)
    
    bt = Backtest(
        stock_data=y, 
        strategy=Strat1, 
        cash=100, 
        commission=.002,
        start_date=datetime.datetime(2007, 4, 1),
        end_date=datetime.datetime(2021, 1, 26),
        )
    bt.start()
    perf = bt.stats.performance
    assets = bt.stats.asset_values
    
    
    
    
    
    