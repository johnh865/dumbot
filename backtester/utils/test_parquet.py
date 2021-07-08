# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np

from backtester.utils.db import ParquetClient


def build_data() -> ParquetClient:
    """Create test dataframes."""
    
    p = ParquetClient('test-dir')
    
    x = np.linspace(0, 1, 50)
    for ii in range(10):
        y = np.sin(x + ii)
        z = np.cos(x + ii)
        d = {}
        d['y'] = y
        d['z'] = z
        df = pd.DataFrame(d, index=x)
        name = 'data-' + str(ii)
        p.save_dataframe(df, name)
    return p


def delete_data(p: ParquetClient):
    p.delete()
    
    
@pytest.fixture(scope='module')
def fixture() -> ParquetClient:
    p = build_data()
    yield p
    p.delete()
    
    
def test_parquet(p : ParquetClient):
    names = p.get_table_names()
    for name in names:
        df = p.read_dataframe(name)
        _, ii = name.split('-')
        ii = int(ii)
        x = df.index
        y = np.sin(x + ii)
        z = np.cos(x + ii)        

        assert np.all(y == df['y'])
        assert np.all(z == df['z'])
        
        
def test_repeat(p: ParquetClient):
    """Repeat the test given already constructed parquet data."""
    return test_parquet(p)
        
        
if __name__ == '__main__':
    p = build_data()
    test_parquet(p)
    test_repeat(p)
    p.delete()