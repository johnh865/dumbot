# -*- coding: utf-8 -*-
import pdb
import pathlib
import shutil
import os

import logging
import sqlite3
import numpy as np
import pandas as pd
from pandas.io.sql import read_sql

from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import inspect

import pyarrow as pa
import pyarrow.parquet as pq


def append_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    # old_index = df1.index.values
    # new_index = df2.index.values

    # ii = np.searchsorted(old_index, new_index)
    # ii = np.minimum(ii, len(old_index) - 1)

    # same_locs = new_index == old_index[ii]
    # diff_locs = ~same_locs
    # df_append = df2.iloc[diff_locs]    
    # df_new = pd.concat([df1, df_append])
    df_new = pd.concat([df1, df2])
    return df_new


def append_to_parquet_table(dataframe, filepath=None, writer=None):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in 
    parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.

    :param dataframe: pd.DataFrame to be written in parquet format.
    :param filepath: target file location for parquet file.
    :param writer: ParquetWriter object to write pyarrow tables in parquet format.
    :return: ParquetWriter object. This can be passed in the subsequent
    method calls to append DataFrame
        in the pyarrow Table
        
        
    https://stackoverflow.com/questions/47113813/using-pyarrow-how-do-you-append-to-parquet-file
    """
    table = pa.Table.from_pandas(dataframe)
    if writer is None:
        writer = pq.ParquetWriter(filepath, table.schema)
    writer.write_table(table=table)
    return writer



class ParquetClient:
    """Construct and modify parquet data."""
    
    suffix = '.parquet'
    
    def __init__(self, directory: str):
        self.dir_path = pathlib.Path(directory)
        if not self.dir_path.exists():
            self.dir_path.mkdir(exist_ok=True)
            
        
    def drop_table(self, name: str):
        path = self.dir_path / name 
        path = path.with_suffix(self.suffix)
        path.unlink()
        return
    
    
    def table_exists(self, name: str) -> bool:
        """Check if table exists"""
        path = self.dir_path / name 
        path = path.with_suffix(self.suffix)    
        return path.is_file()
    
    
    def get_table_names(self):
        paths = self.dir_path.iterdir()
        stems = [p.stem for p in paths if p.suffix == self.suffix]
        return stems
    
    
    def save_dataframe(self, df: pd.DataFrame, name: str):
        path = self.dir_path / name
        basedir = path.parent
        os.makedirs(basedir, exist_ok=True)
        
        path = path.with_suffix(self.suffix)
        df.to_parquet(str(path), )


    def read_dataframe(self, name: str):
        """Read from database to dataframe."""
        path = self.dir_path / name
        path = path.with_suffix(self.suffix)
        df = pd.read_parquet(path) 
        return df
    
    
    def append_dataframe(self, df: pd.DataFrame, name: str):
        """Append dataframe to database."""
        df_old = self.read_dataframe(name)
        df_new = append_dataframes(df_old, df)
        self.save_dataframe(df_new, name)
        return df_new
    
    
    def delete(self):
        """Delete all stored data."""
        # paths = self.dir_path.iterdir()
        # for path in paths:
        #     path.unlink()
        # self.dir_path.unlink()
        shutil.rmtree(str(self.dir_path), )
        
        
    
    


class SQLClient:
    def __init__(self, url, **kwargs):
        self.engine = create_engine(url, **kwargs)
        
        
    def connect(self):
        return self.engine.connect()
        
        
    def drop_table(self, name: str):
        table_name = name
        engine = self.engine
        base = declarative_base()
        metadata = MetaData(engine)
        metadata.reflect(bind=engine)
        table = metadata.tables.get(table_name)
        if table is not None:
            logging.info(f'Deleting {table_name} table')
            base.metadata.drop_all(engine, [table], checkfirst=True)
            
            
            
    def get_table_names(self):
        engine = self.engine
        insp = inspect(engine)
        return insp.get_table_names()
    
    
    def save_dataframe(self, df: pd.DataFrame, name: str):
        """Save dataframe to database."""
        df.to_sql(name=name,
                  con=self.engine,
                  if_exists='replace',
                  method='multi')
        
        
    def append_dataframe(self, df: pd.DataFrame, name: str):
        """Append dataframe to database."""
        df_old = self.read_dataframe(name)
        old_index = df_old.index.values
        new_index = df.index.values
        
        ii = np.searchsorted(old_index, new_index)
        ii = np.minimum(ii, len(old_index)-1)
        
        same_locs = new_index == old_index[ii]
        diff_locs = ~same_locs
        df_append = df.iloc[diff_locs]
        df_append.to_sql(name=name,
                         con=self.engine,
                         if_exists='append',
                         method='multi')
        
        df_new = pd.concat([df_old, df_append])
        return df_new
        
        
    def read_dataframe(self, name: str):
        """Read from database to dataframe."""
        df = pd.read_sql(name, self.engine, index_col=0)
        index_col = df.columns.values[0]
        df = df.set_index(index_col)
        return df
    
    
    def rename_table(self, old: str, new: str):
        """Rename a table."""
        df = self.read_dataframe(old)
        self.drop_table(old)
        self.save_dataframe(df, new)
        
        
    def to_memory(self):
        """Save db to sqlite in memory"""
        engine = create_engine('sqlite://')
        c = engine.connect()
        # con_mem = c.connection
        table_names = self.get_table_names()
        
        #Here is the connection to <ait.sqlite> residing on disk
        # con = sqlite3.connect(name, isolation_level=None)
        # cur = con.cursor()
        for table in table_names:
            df = pd.read_sql(table, engine)
            df.to_sql(con=engine,
                      name=table,
                      if_exists='replace')
            
        return engine
    
    
            


        
        


