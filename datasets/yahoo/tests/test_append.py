from datasets.yahoo.build_db import update

from datasets import symbols
from datasets.yahoo.definitions import CONNECTION_PATH, TABLE_SYMBOL_PREFIX

from backtester import utils

SYMBOLS = symbols.ALL


def test_update():
    update(symbols.ALL[0:10])


test_update()