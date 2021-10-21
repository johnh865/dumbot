# -*- coding: utf-8 -*-
"""Attempt to clean up database."""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from datasets.yahoo.read import read_yahoo_symbol_names, read_yahoo_dataframe
from datasets.yahoo.definitions import (
    DF_ADJ_CLOSE, CONNECTION_PATH, TABLE_GOOD_SYMBOL_DATA
    )



def _array_jumps(arr : np.ndarray, maxsize=1.9) -> bool:
    """Determine if series of data has a weird jump in price."""
    values = arr
    v0 = values[0: -1]
    v1 = values[1:]
    delta = v1 - v0
    ratio1 = delta / v0
    ratio2 = delta / v1
    if np.any(ratio1 > maxsize):
        return True
    if np.any(ratio2 > maxsize):
        return True


def has_jumps(series: pd.Series, maxsize=1.9) -> bool:
    """Determine if series of data has a weird jump in price."""
    return _array_jumps(series.values, maxsize=maxsize)
  
  
def has_gaps(series: pd.Series, maxsize=10) -> bool:
    """Determine if series of data has weird time gaps."""
    dates = (series.index
                   .values
                   .astype('datetime64[D]')
                   .astype(float))
    return _array_jumps(dates, maxsize=maxsize)


def min_points(series: pd.Series, ptnum=200) -> bool:
    """Demand a minimum number of points."""
    values = series.values
    isnan = np.isnan(values)
    values = values[~isnan]
    return len(values) < ptnum
    


def is_clean_symbol(df: pd.DataFrame, name='') -> bool:
    closes = df[DF_ADJ_CLOSE]
        
    if len(df) == 0:
        print(name, 'has no data in it')
    elif has_jumps(closes):
        print(name, 'has data jumps.')
    elif has_gaps(closes):
        print(name, 'has gaps')
    elif min_points(closes):
        print(name, 'less than 200 points')
    else:
        print(name, '-- GOOD.')
        return True
    return False
    




def get_good_symbols() -> list[str]:
    """Retrieve symbols where I think it has 'good' data."""
    names = read_yahoo_symbol_names()
    good_list = []
    for name in names:
        df = read_yahoo_dataframe(name)
        closes = df[DF_ADJ_CLOSE]
        
        if len(df) == 0:
            print(name, 'has no data in it')
        elif has_jumps(closes):
            print(name, 'has data jumps.')
        elif has_gaps(closes):
            print(name, 'has gaps')
        elif min_points(closes):
            print(name, 'less than 200 points')
        else:
            print(name, '-- GOOD.')
            good_list.append(name)
            
    gnum = len(good_list)
    print(f'{gnum} # of symbols considered "GOOD".')
    return good_list
    
    

def save_good_table():
    """Save TABLE_GOOD_SYMBOL_DATA to database."""
    engine = create_engine(CONNECTION_PATH, echo=False)
    good_list = get_good_symbols()
    
    (pd.Series(good_list, name='symbol')
         .to_frame()
         .to_sql(name=TABLE_GOOD_SYMBOL_DATA,
                 con=engine,
                 if_exists='replace'))




if __name__ == '__main__':
    save_good_table()


    