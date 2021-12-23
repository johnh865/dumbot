# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np
import pdb

from backtester.utils.db import ParquetClient, append_to_parquet_table


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
        print(f'Creating dataframe "{name}"')
    return p


def delete_data(p: ParquetClient):
    """Delete data for cleanup after test."""
    print('Deleting parquet data.')
    p.delete()
    
    
@pytest.fixture(scope='module')
def fixture() -> ParquetClient:
    p = build_data()
    yield p
    
    print('Cleaning up')
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



# def test_append(p: ParquetClient):
#     """Test the append function."""
#     x = np.linspace(1.2, 2, 30)
#     d = {}
#     d['y'] = x * 4
#     d['z'] = x * 10
#     df = pd.DataFrame(d, index=x)
    
#     target = 'data-0'
#     print(f'appending to {target}')
#     path = p.dir_path / (target + '.parquet')
#     append_to_parquet_table(df, filepath=path)
#     df = p.read_dataframe(target)
    
#     assert df.index[-1] == 2.0
#     assert df['y'].values[-1]== d['y'][-1]
#     pdb.set_trace()

#     return
    
    
        
        
if __name__ == '__main__':
    p = build_data()
    test_parquet(p)
    test_repeat(p)
    # test_append(p)
    p.delete()