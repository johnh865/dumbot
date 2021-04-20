from os.path import dirname, join


PACKAGE_PATH = dirname(__file__)
PROJECT_PATH = dirname(PACKAGE_PATH)
CONNECTION_PATH = 'sqlite:///' + join(PACKAGE_PATH, 'db.sqlite3')

DF_DATE = 'Date'
DF_ADJ_CLOSE = 'Adj Close'
DF_HIGH = 'High'
DF_LOW = 'Low'
DF_CLOSE = 'Close'
DF_OPEN = 'Open'
DF_VOLUME = 'Volume'


# Table name for avaible trade dates
TABLE_ALL_TRADE_DATES = 'all-trade-dates'
TABLE_GOOD_SYMBOL_DATA = 'good-symbol-list'

TABLE_SYMBOL_PREFIX = 'symbol-'