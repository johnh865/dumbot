# -*- coding: utf-8 -*-
import pdb
import numpy as np



# Parse the given data file. 
CSV_FILE = 'S&P 500 Historical Components & Changes(03-01-2021).csv'


def read_csv_sp500():
    with open(CSV_FILE, 'r') as f:
        lines = f.readlines()
        
        out = {}
        for ii, line in enumerate(lines[1:]):
            date, tickers = line.split(sep=',', maxsplit=1)
            # year, day, month = date.split('-')
            
            date = np.datetime64(date)
            # print(ii, year)
            tickers = tickers.strip()
            tickers = tickers.replace('"', '')
            ticker_list = tickers.split(',')
            out[date] = ticker_list
    return out
    
    

DATA_DICT = read_csv_sp500()
DATA_DATES = np.array(list(DATA_DICT.keys()))



def get500(date: np.datetime64) -> list[str]:
    """Return list of SP500 symbols"""
    try:
        return DATA_DICT[date]
    except KeyError:
        ii = np.searchsorted(DATA_DATES, date)
        date_ii = DATA_DATES[ii - 1]
        
        assert date_ii < date
        
        return DATA_DICT[date_ii]
    
    

def test1():
    date = np.datetime64('2012-01-01')
    get500(date)
    
    
    
if __name__ == '__main__':
    test1()