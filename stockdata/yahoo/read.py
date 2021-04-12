# -*- coding: utf-8 -*-
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy import inspect

import pandas as pd

from stockdata.symbols import ALL
from stockdata.yahoo.definitions import CONNECTION_PATH
from stockdata.yahoo.definitions import (
    DF_DATE,
    TABLE_ALL_TRADE_DATES,
    TABLE_SYMBOL_PREFIX,
    )


engine = create_engine(CONNECTION_PATH, echo=False)

@lru_cache(maxsize=500)
def read_yahoo_dataframe(symbol: str) -> pd.DataFrame:
    """Read all available stock symbol Yahoo data."""
    dataframe = pd.read_sql(symbol, engine).set_index(DF_DATE, drop=True)
    return dataframe
	

def read_yahoo_tablenames() -> list[str]:
    insp = inspect(engine)
    return insp.get_table_names()


def read_yahoo_trade_dates() -> pd.DataFrame:
    df = pd.read_sql(TABLE_ALL_TRADE_DATES, engine)
    return df[DF_DATE].values