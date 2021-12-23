# -*- coding: utf-8 -*-



from datasets.yahoo2.downloader import YahooClient
from datasets.yahoo2.definitions import PARQUET_PATH
from datasets.symbols import ALL as ALL_SYMBOLS



y = YahooClient(PARQUET_PATH, ALL_SYMBOLS)
y.update()