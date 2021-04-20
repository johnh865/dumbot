# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import backtester

from sqlalchemy import create_engine


from backtester.indicators import TrailingStats
from backtester.analysis import BuySell, avg_future_growth
from backtester.stockdata import YahooData, read_yahoo_symbol_names
from datasets.symbols import ALL
from backtester.definitions import DF_ADJ_CLOSE
from backtester import utils


from dumbot.definitions import CONNECTION_PATH

def create_data(symbols: list[str]):
    # np.random.seed(0)
    # symbols = np.array(ALL)
    # np.random.shuffle(symbols)
    # symbols = symbols[0:1]
    
    y = YahooData(symbols)
    
    # window_sizes = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 400]
    
    window_sizes = [5, 10, 20, 50, 100, 400]
    window_sizes = np.array(window_sizes)
    max_window = np.max(window_sizes)
    future_growth_window = 20
    attrs = ['exp_growth', 'rolling_avg',]
    
    df_dict = {}
    for ii, symbol in enumerate(symbols):
        print(f'Calculating symbol {symbol}')

        df = y.get_symbol_all(symbol)
        if len(df) == 0:
            print(f'Symbol {symbol} data not available')
        else:            
            series = df[DF_ADJ_CLOSE]
            new = {}
            
            for window in window_sizes:
                
                # Get indicators
                ts = TrailingStats(series, window)
                for attr in attrs:
                    feature = getattr(ts, attr)
                    name = attr + f'({window})'
                    new[name] = feature[0 : -future_growth_window]
                    
            
            # Get future growths
            times = utils.dates2days(series.index.values)
            dates = series.index[0 : -future_growth_window]
        
            _, growths = avg_future_growth(times,
                                           series.values, 
                                           window=future_growth_window)
            
            new['avg_future_growth'] = growths
            new['date'] = dates
            
            df_new = pd.DataFrame(new)
            
            # Chop of NAN due to the largest window
            df_new = df_new.iloc[max_window :]
            df_dict[symbol] = df_new
    return df_dict
        
    
def save_db(df_dict: dict):
    engine = create_engine(CONNECTION_PATH, echo=False)
    
    df: pd.DataFrame
    for key, df in df_dict.items():
        name = 'symbol-' + key
        df.to_sql(name, con=engine, if_exists='append')
    return


def load_db(symbols: list[str]):
    engine = create_engine(CONNECTION_PATH, echo=False)
    
    df_dict = {}
    for symbol in symbols:
        name = 'symbol-' + symbol
        df = pd.read_sql(name, engine)
        df_dict[symbol] = df
    return df_dict

def load_symbol_names():
    
    table_names = utils.get_table_names(CONNECTION_PATH)
    return [s.replace('symbol-', '') for s in table_names]


if __name__ == '__main__':
    np.random.seed(0)
    symbols = read_yahoo_symbol_names()

    np.random.shuffle(symbols)
    symbols = symbols[0:10]
    df_dict = create_data(symbols)
    save_db(df_dict)
    