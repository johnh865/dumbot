# -*- coding: utf-8 -*-

import pdb
import numpy as np
import pandas as pd
import pytest 

from backtester.backtest import Strategy, Backtest
from backtester.stockdata import YahooData, Indicators, DictData
from backtester.indicators import TrailingStats, append_nan
from backtester.definitions import DF_ADJ_CLOSE

symbols = ('SPY', 'GOOG', 'MSFT')
indicator_keys = ['RoR-10', 'RoR-20', 'RoR-30']
yahoo = YahooData(symbols=symbols)
                  
                  
                  
def indicator10(df: pd.DataFrame, window_size=10):
    """Indicator construction function."""
    series = df[DF_ADJ_CLOSE]
    ts = TrailingStats(series, window_size)
    y = ts.return_ratio
    x = ts.times
    new_series = pd.Series(y, index=x, name=f'RoR-{window_size}')
    new_df = pd.DataFrame(new_series)
    return new_df


def build_dict_data_20(window_size=20):
    """Create DictData indicator for 'RoR-20'."""
    d = {}
    for key, df in yahoo.dataframes.items():
        d[key] = indicator10(df, window_size=window_size)
    return DictData(d)
    
    
def build_column_data_30():
    """Create DataFrame of indicator 'RoR-30'."""
    dictdata = build_dict_data_20(30)
    symbol = dictdata.symbol_names[0]
    column = dictdata.tables[symbol].columns[0]
    return dictdata.extract_column(column)
        
        
        
class Strategy1(Strategy):
    """Dummy strategy that only constructs indicators."""
    def init(self):
        dd = build_dict_data_20()
        cd = build_column_data_30()        
            
        self.indicators.from_func(indicator10, key='RoR-10')
        self.indicators.from_basedata(dd, key='RoR-20')
        self.indicators.from_dataframe(cd, key='RoR-30')
        self.outputs = {}
        
    def next(self):
        output = self.indicators.get_latest()
        self.outputs[self.date] = output
    
    
def run_backtest() -> Backtest:
         
    date1 = np.datetime64('2016-01-01')
    date2 = np.datetime64('2016-01-20')
    bt = Backtest(
        stock_data=yahoo,
        strategy=Strategy1,
        start_date=date1,
        end_date=date2,
        )
    bt.run()
    return bt


@pytest.fixture(scope='module')
def run_backtest_fixture():
    return run_backtest()
    

def test_create_data():
    """Test creation of dict and column data."""
    dd = build_dict_data_20()
    cd = build_column_data_30()    
    assert isinstance(dd, DictData)
    assert isinstance(cd, pd.DataFrame)
    

def test_backtest(run_backtest_fixture: Backtest):
    """Test indicator construction in backtesting."""
    bt = run_backtest_fixture
    
    assert np.all(bt.strategy.indicators.keys() == indicator_keys)
    
    
def test_latest(run_backtest_fixture: Backtest):
    bt = run_backtest_fixture
    outputs = bt.strategy.outputs
    for date, df in outputs.items():
        assert np.all(df.index == indicator_keys)
        assert np.all(df.columns.values.astype(str) == symbols)
        
        
if __name__ == '__main__':
    test_create_data()
    
    bt = run_backtest()
    test_backtest(bt)
    test_latest(bt)
    