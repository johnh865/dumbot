# -*- coding: utf-8 -*-

"""Get list of stock symbol strings."""
from datasets.definitions import PACKAGE_PATH
from os.path import join

__all__ = ['SP500', 'FUNDS', 'ALL']

with open(join(PACKAGE_PATH, 'sp500', 'symbol_list.txt')) as f:
    _string = f.read()
    SP500 = _string.split()

with open(join(PACKAGE_PATH, 'funds_list.txt')) as f:
    _string = f.read()
    FUNDS = _string.split()
    


ALL = []
ALL.extend(SP500)
ALL.extend(FUNDS)