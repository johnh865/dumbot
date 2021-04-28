# -*- coding: utf-8 -*-


import logging
import sqlite3
import pandas as pd
from pandas.io.sql import read_sql

from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import inspect


class SQLClient:
    def __init__(self, *args, **kwargs):
        self.engine = create_engine(*args, **kwargs)
        
        
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
        df.to_sql(name=name, con=self.engine)
        
        
    def read_dataframe(self, name: str):
        """Read from database to dataframe."""
        return pd.read_sql(name, self.engine)
    
    
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
    
    
            


        
        


