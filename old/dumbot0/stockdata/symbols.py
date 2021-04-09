# -*- coding: utf-8 -*-

"""Get list of stock symbol strings."""
from dumbot.definitions import PACKAGE_PATH
from os.path import join
with open(join(PACKAGE_PATH, 'stockdata', 'symbol_list.txt')) as f:
    _string = f.read()
    SYMBOLS = _string.split()

with open(join(PACKAGE_PATH, 'stockdata', 'funds_list.txt')) as f:
    _string = f.read()
    FUNDS = _string.split()
    


ALL = []
ALL.extend(SYMBOLS)
ALL.extend(FUNDS)