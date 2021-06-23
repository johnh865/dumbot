# -*- coding: utf-8 -*-
import pdb

import logging
import sqlite3
import numpy as np
import pandas as pd
from pandas.io.sql import read_sql

from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import inspect


class SQLClient:
    def __init__(self, url, **kwargs):
        self.engine = create_engine(url, **kwargs)
        
        
    def connect(self):
        return self.engine.connect()
        
        
    def drop_table(self, table_name: str):
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
        df.to_sql(name=name, con=self.engine, if_exists='replace')
        
        
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
        df_append.to_sql(name=name, con=self.engine, if_exists='append')
        
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
    
    
            


        
        


