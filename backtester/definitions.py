# -*- coding: utf-8 -*-
from os.path import dirname, join


CONNECTION_NAME = 'sqlite:///db.sqlite3'
PACKAGE_PATH = dirname(__file__)
PROJECT_PATH = dirname(PACKAGE_PATH)
CONNECTION_PATH = 'sqlite:///' + join(PACKAGE_PATH, 'data', 'db.sqlite3')
DATA_PATH = join(PACKAGE_PATH, 'data')

# Define a 'small number' for round-off calcs. 
SMALL_DOLLARS = 1e-4


DF_DATE = 'Date'
DF_ADJ_CLOSE = 'Adj Close'
DF_HIGH = 'High'
DF_LOW = 'Low'
DF_CLOSE = 'Close'
DF_OPEN = 'Open'
DF_VOLUME = 'Volume'

DF_SMOOTH_CLOSE = 'Smooth Adj Close'
DF_SMOOTH_CHANGE = 'Smooth Rel Change'
DF_TRUE_CHANGE = 'Rel Change'



ACTION_BUY = 'buy'
ACTION_SELL = 'sell'
ACTION_DEPOSIT = 'deposit'
ACTION_WITHDRAW = 'withdraw'
ACTION_HOLD = 'hold'
ACTION_SELL_PERCENT= 'sell_percent'

