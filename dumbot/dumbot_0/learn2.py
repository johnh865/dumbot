# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dumbot.definitions import CONNECTION_PATH
from dumbot.build_data2 import load_db

# %% load & process features & targets

def load():
    symbols = ['GOOG']
    d = load_db(symbols)
    df = pd.concat(d)
    return df


def process(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Extract features and targets for ML."""
        
    dates = df['date']
    date_min = dates.min()
    time_delta = dates - dates.min()
    
    col_names = df.columns
    feature_names = [c for c in col_names if c.endswith(')')]
    features = df[feature_names]
    targets = df['avg_future_growth']
    return features, targets



df = load()
features, targets = process(df)



# p01 = np.percentile(targets, 1)
# p99 = np.percentile(targets, 99)

# %%

from sklearn import linear_model, svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression

scaler_x = preprocessing.StandardScaler().fit(features.values)
# scaler_y = preprocessing.StandardScaler().fit(targets.values[:, None])

features1 = scaler_x.transform(features)
# targets1 = scaler_y.transform(targets.values[:, None]).ravel()

splitter = TimeSeriesSplit()

x_train, x_test, y_train, y_test = train_test_split(features1, 
                                                    targets, 
                                                    test_size=0.4,
                                                    random_state=0)
# reg = linear_model.LinearRegression()
# reg.fit(features, targets)

# regr = svm.SVR(cache_size=4000)
# regr.fit(x_train, y_train)

# score = regr.score(x_test, y_test)


# estimators = np.arange(10, 1000, 10)
# scores = []
# for estimator in estimators:
# reg = RandomForestRegressor()
reg = LinearRegression()
reg.fit(x_train, y_train)


score = reg.score(x_train, y_train)
print('train score', score)
y_predict = reg.predict(x_test)

score = reg.score(x_test, y_test)
print('test score', score)
    

plt.plot(y_predict, y_test, '.', alpha=.2)
plt.grid()
plt.xlabel('Prediction')
plt.ylabel('True')
