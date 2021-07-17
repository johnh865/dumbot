# -*- coding: utf-8 -*-
import pdb
from functools import cached_property

import numpy as np
import pandas as pd

from backtester.backtest import Strategy, Backtest
from backtester.model import TransactionsLastState, MarketState
from backtester.model import SymbolTransactions, Action
from backtester.model import ACTION_BUY, ACTION_SELL_PERCENT, ACTION_SELL

import matplotlib.pyplot as plt
import seaborn as sns


from bokeh.plotting import figure, show
from bokeh.palettes import RdYlBu, inferno
from bokeh.embed import file_html, components
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker
from bokeh.transform import factor_cmap

__all__ = ['barplot']

TRADING_DAYS_PER_YEAR = 253 

  
    
def barplot(df: pd.DataFrame,
            x: str,
            y: str,
            c: str=None,       
            tooltipcols: list[str]=None,
            width=650, height=400, title='', fmt='0.2f'):
    df = df.copy()
    xvals = df[x]
    yvals = df[y]
    
    
    # Create formatted text labels
    texts = []
    text_fmt = ' {xi:' + fmt + '}  '
    for xi in xvals:
        texts.append(text_fmt.format(xi=xi))
    text_name = '__texts__'
    df[text_name] = texts
    
    height = len(yvals) * 25 + 60
    
    
    # Format tooltips
    if tooltipcols is not None:
        tooltips = []
        for column in tooltipcols:
            elem = (column, '@{' + column + '}')
            tooltips.append(elem)
    else:
        tooltips = None
        
        
    # Get colors
    if c is not None:
        cvals = df[c]
        mapper = LinearColorMapper(
            palette=list(reversed(inferno(20))), 
            low=cvals.min(),
            high=cvals.max(),
            )
        fill_color = {
            'field' : c,
            'transform' : mapper
            }      
        color_bar = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(desired_num_ticks=10)
            )
        
    else:
        fill_color = 'blue'
        color_bar = None
    
    fig = figure(
        y_range = yvals,
        plot_height = height,
        plot_width = width,
        title = title,
        tools = 'save',
        tooltips = tooltips,
        )
    fig.x_range.end = xvals.max()*1.1
    fig.hbar(
        y=y,
        right=x,
        source=df,
        height=0.85,
        fill_color=fill_color
        )
    fig.text(
        source=df,
        y=y,
        x=x,
        text=text_name,
        text_align='left',
        text_baseline='middle',
        text_font_size='12px',
        color='#000000')
    
    
    if color_bar:
        fig.add_layout(color_bar, 'right')    
        
    fig.xaxis.axis_label_text_font_size = '12pt'
    fig.xaxis.axis_label = x

    fig.xaxis.major_label_text_font_size = '12pt'
    fig.yaxis.axis_label_text_font_size = '12pt'
    fig.yaxis.major_label_text_font_size = '12pt'   
    fig.title = title
    # html = file_html(plot, CDN, title=title)
    return fig
    





        