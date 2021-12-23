# -*- coding: utf-8 -*-

"""Retrieve symbols in S&P500. Let's filter since the year 2000.

Taken from github:
https://github.com/fja05680/sp500"""

import pdb 

# Parse the given data file. 
fname = 'S&P 500 Historical Components & Changes(03-01-2021).csv'

with open(fname, 'r') as f:
    lines = f.readlines()

ALL = set()

for ii, line in enumerate(lines[1:]):
    date, tickers = line.split(sep=',', maxsplit=1)
    year, day, month = date.split('-')
    # print(ii, year)
    if int(year) >= 2000: 
        tickers = tickers.strip()
        tickers = tickers.replace('"', '')
        ticker_list = tickers.split(',')
    
        ALL.update(ticker_list)

ALL = list(ALL)
ALL.sort()
        
fname2 = 'symbol_list.txt'
with open(fname2, 'w') as f:
    string = '\n'.join(ALL)
    f.write(string)