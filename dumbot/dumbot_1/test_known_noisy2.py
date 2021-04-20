# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    
    
    

class CreateData:
    def __init__(self, series: pd.Series):
        self.series = series
            
            
    def features(self, window: int):
        attrs = ['exp_growth', 'exp_accel']
        # attrs = ['exp_accel']
        series1 = self.series
        ts1 = TrailingStats(series1, window)

        new = {}
        for attr in attrs:
            feature = getattr(ts1, attr)
            name = f'Close({attr}, {window})'
            new[name] = feature

        df_new = pd.DataFrame(new, index=series.index)
        df_new = df_new.iloc[window :]
        return df_new

        
    def targets(self, window: int):
        
        # Get future growths
        series1 = self.series
        times = series1.index.values
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


# %% Build data

ptnum = 10000
x0 = np.linspace(0, 30, ptnum)
y0 = np.sin(x0) + x0/2 + 10

# %% Add Noise

rstate = np.random.default_rng(seed=0)
r_amp = .000005
r = r_amp * rstate.normal(size=ptnum)
rwalk = np.cumsum(r)

# Add "corrective pressure to the rwalk
# r_amp2 = .03
# r_correct = -rstate.integers(-1, 0, size=ptnum) * np.sign(rwalk)
# r2 = rstate.normal(size=ptnum) * r_correct * r_amp2
# rwalk2 = np.cumsum(r2)

y0r = y0 + rwalk
# y0r2 = y0 + rwalk + rwalk2





plt.figure()
plt.plot(x0, y0, label='true')
plt.plot(x0, y0r, label='noised')
# plt.plot(x0, y0r2, label='noised-corrected')
plt.plot(x0, rwalk, label='walk')
plt.legend()

# %% Build metrics

series = pd.Series(y0r, index=x0)

ts = TrailingStats(series, 10)

creator = CreateData(series)

windows1 = [500, 1000, 2000]
y = creator.targets(500)

xlist = []
for window in windows1:
    print(f'building window {window}')
    x = creator.features(window)
    xlist.append(x)
x1, y1 = combine_datas(xlist, y) 

plt.figure()
plt.subplot(2,2,1)
plt.plot(x0, y0, label='true')
plt.plot(x0, y0r, label='noised')
plt.plot(x0, rwalk, label='walk')
plt.legend()


plt.subplot(2,2,2)
plt.title('Targets')
plt.plot(y1, label='target')


for name in x1:
    if 'exp_growth' in name:
        plt.subplot(2,2,3)
    elif 'accel' in name:
        plt.subplot(2,2,4)
    plt.plot(x1[name], label=name)
plt.legend()
plt.title('growth')

plt.subplot(2,2,3)
plt.legend()
plt.title('accel')

# %% Training

split_index = 7000
x_train = x1.values[1 : split_index]
y_train = y1.values[1 : split_index]
x_test = x1.values[split_index :]
y_test = y1.values[split_index :]    
    




clf = RandomForestRegressor()
clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)
y_predict = clf.predict(x_test)
plt.figure()
plt.plot(y_predict, y_test, '.')
print('score', score)