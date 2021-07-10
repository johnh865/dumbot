# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class Cell:
    def __init__(self, text: str):
        self.text = text
        
    def __str__(self):
        out = '''\
            test text'''
        
        
        return out
    
    
        

class Table:
    def __init__(self, df: pd.DataFrame):
        pass
    
    
# class Figure:
#     def __init__
    
class JupyterReport:
    def __init__(self, path: str):
        self.path = path
        self._tables = []
        self._figures = []
        
        
    def table(self, df: pd.DataFrame):
        pass
    
    
    def figure(self, ax):
        pass
    
    
    def save(self):
        with open(self.path, 'w') as f:
            pickle.dump(self, f)
            
            
            
    def load(self):
        pass
    
    
        
    
    
    
        
        
    
        