# -*- coding: utf-8 -*-
import pdb
import datetime
import numpy as np
import pandas as pd
    
def get_trading_days(dates : np.ndarray,
                     date1 : datetime.date, 
                     date2 : datetime.date=None):
    """For two dates, get trading days in between."""
    date1 = np.datetime64(date1)
    date2 = np.datetime64(date2)
    
    if not np.isnat(date1):
        dates = dates[dates >= date1]

    if not np.isnat(date1):
        dates = dates[dates <= date2]
    return dates


    
def dates2days(dates: np.ndarray):
    """Convert np.datetime64 to days from start as array[int] ."""
    tdelta = dates - dates[0]
    tdelta = tdelta.astype('timedelta64[D]')
    return tdelta.astype(int)


def datetime_np2py(date: np.datetime64):
    t = np.datetime64(date, 'us').astype(datetime.datetime)
    return t



def datetime_to_np(date: datetime.datetime) -> np.array:
    """Convert a time object/array to numpy datetime64, or numpy array."""
    
    # Try to convert a scalar time
    try:
        out = np.datetime64(date)
        
    # Try to convert array time
    except ValueError:
        out = np.asarray(date).astype('datetime64')
    return out
    


def floor_to_date(date: datetime.datetime):
    
    try:
        date = date.replace(hour=0, minute=0, second=0, microsecond=0)    
    except AttributeError:
        date = np.datetime64(date, 'D')        
    return date


def floor_dates(dates: np.ndarray):
    """Floor dates to day."""
    return np.asarray(dates).astype('datetime64[D]')





EPOCH = np.datetime64('1970-01')
EPOCH_YR = np.datetime64('0')

def get_year(dates: np.ndarray):
    dates = dates.astype('datetime64[Y]')
    delta = dates - EPOCH_YR
    return delta.astype(int)


def get_quarters(dates: np.ndarray):    
    # dates = dates.astype('datetime64[M]')
    # delta = dates - EPOCH + 1
    # quarters = delta.astype(int) // 3 % 4 + 1
    return pd.DatetimeIndex(dates).quarter.values
    


def round_to_quarters(dates: np.ndarray) -> np.ndarray:
    year = get_year(dates) + EPOCH_YR
    quarters = get_quarters(dates)
    months = quarters * 3 - 2
    date = year.astype('datetime64[Y]').astype('datetime64[M]')
    
    return date + months - 1
    
    









    
    