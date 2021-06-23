# -*- coding: utf-8 -*-

import pandas as pd 

def extract_column(d: dict[pd.DataFrame], column: str) -> pd.DataFrame:
    """From all symbols retrieve the specified data column"""
    
    new = []
    for symbol in d:
        df = d[symbol]

        if df.size == 0:
            pass
        else:
            series = df[column]
            series.name = symbol
            new.append(series)
                    
    df = pd.concat(new, axis=1, join='outer',)
    return df


