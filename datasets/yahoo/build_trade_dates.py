# -*- coding: utf-8 -*-

"""
Get valid trade dates from Yahoo data
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from datasets.yahoo import read_yahoo_dataframe
from datasets.symbols import SP500, FUNDS
from datasets.yahoo.definitions import (
    DF_DATE, CONNECTION_PATH, TABLE_ALL_TRADE_DATES
    )

# from backtester.data.symbols import ALL, FUNDS
# from backtester.definitions import (DF_ADJ_CLOSE, 
#                                 DF_DATE,
#                                 TABLE_ALL_TRADE_DATES,
#                                 CONNECTION_PATH,
#                                 )




def _build_trade_dates():
    
    time_old : pd.DataFrame = None
    time_new : pd.DataFrame
    
    np.random.seed(0)
    funds = FUNDS.copy()
    np.random.shuffle(funds)
    

    symbols = ['DIS', 'GOOG'] + funds[0 : 15]
    
    for symbol in symbols:

        time_new = read_yahoo_dataframe(symbol).index.values
        
        if time_old is not None:
            time_old = np.append(time_old, time_new)
            time_old = np.unique(time_old)
            # time2  = time_old.merge(time_new, how='outer',)
            # time_old = time2
            pass
        else:
            time_old = time_new.copy()
    return time_old


def trade_dates_dataframe():
    times = _build_trade_dates()
    series = pd.Series(times)
    df = series.to_frame(DF_DATE)
    return df
    
    
def save_trade_dates_to_sql():
    """Save TABE_ALL_TRADE_DATES to database."""
    times = _build_trade_dates()
    series = pd.Series(times)
    df = series.to_frame(DF_DATE)
    engine = create_engine(CONNECTION_PATH, echo=False)
    df.to_sql(name=TABLE_ALL_TRADE_DATES, con=engine, if_exists='replace')
    return series
    

if __name__ == '__main__':
    pass        
            