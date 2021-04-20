# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

from backtester.utils import get_table_names
from dumbot.definitions import CONNECTION_PATH
from dumbot.build_data import load_db, load_symbol_names

# %% load & process features & targets

def load():
    symbols = load_symbol_names()
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

# %% Begin ML

from sklearn import linear_model, svm
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


scaler_x = preprocessing.StandardScaler().fit(features.values)
# scaler_y = preprocessing.StandardScaler().fit(targets.values[:, None])

features1 = scaler_x.transform(features)
# targets1 = scaler_y.transform(targets.values[:, None]).ravel()

# %%
x_train, x_test, y_train, y_test = train_test_split(features1, 
                                                    targets, 
                                                    test_size=0.9,
                                                    random_state=0)
# reg = linear_model.LinearRegression()
# reg.fit(features, targets)

# regr = svm.SVR(cache_size=4000)
# regr.fit(x_train, y_train)

# score = regr.score(x_test, y_test)

reg = MLPRegressor(hidden_layer_sizes=[100]*20, )
reg.fit(x_train, y_train)
# score = regr.score(x_test, y_test)

# %%

plt.plot(reg.loss_curve_)

# %%

from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

y_pred = reg.predict(x_test)
plt.plot(y_test, y_pred, '.', alpha=.1)




