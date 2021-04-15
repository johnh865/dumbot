# -*- coding: utf-8 -*-
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy import inspect

import pandas as pd

from datasets.symbols import ALL
from datasets.yahoo.definitions import CONNECTION_PATH
from datasets.yahoo.definitions import (
    DF_DATE,
    TABLE_ALL_TRADE_DATES,
    TABLE_SYMBOL_PREFIX,
    )


engine = create_engine(CONNECTION_PATH, echo=False)

def read_yahoo_dataframe(symbol: str) -> pd.DataFrame:
    """Read all available stock symbol Yahoo data."""
    name = TABLE_SYMBOL_PREFIX + symbol
    dataframe = pd.read_sql(name, engine).set_index(DF_DATE, drop=True)
    return dataframe
	

def read_yahoo_symbol_names() -> list[str]:
    insp = inspect(engine)
    tables = insp.get_table_names() 
    new = []
    for table_name in tables:
        if table_name.startswith(TABLE_SYMBOL_PREFIX):
            new.append(table_name.replace(TABLE_SYMBOL_PREFIX, ''))
    return new


def read_yahoo_trade_dates() -> pd.DataFrame:
    df = pd.read_sql(TABLE_ALL_TRADE_DATES, engine)
    return df[DF_DATE].values