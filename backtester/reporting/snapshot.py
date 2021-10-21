# -*- coding: utf-8 -*-
import pdb
from functools import cached_property
import copy
import numpy as np
import pandas as pd

from backtester.backtest import Strategy, Backtest
from backtester.model import TransactionsLastState, MarketState
from backtester.model import SymbolTransactions, Action
from backtester.model import ACTION_BUY, ACTION_SELL_PERCENT, ACTION_SELL

from backtester.reporting.plots import barplot

import matplotlib.pyplot as plt
import seaborn as sns


from bokeh.plotting import figure, show
from bokeh.palettes import RdYlBu, inferno
from bokeh.embed import file_html, components
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker
from bokeh.transform import factor_cmap
from bokeh.layouts import column


TRADING_DAYS_PER_YEAR = 253 


class SymbolReport:
    def __init__(self,
                 symbol_transactions: SymbolTransactions,
                 date: np.datetime64=None):
        self.transactions = symbol_transactions
        self.actions = symbol_transactions.executed_actions
        self.name = self.transactions.symbol

        
    @cached_property
    def hold_locations(self) -> np.ndarray:
        """ndarray[bool] : Action index locations where
        shares are greater than zero."""
        shares = [action.shares for action in self.actions]
        shares = np.array(shares)
        return shares > 0
    
    
    @cached_property
    def buy_locs(self):
        """Action index locations of purchases."""
        locations = []
        for ii, action in enumerate(self.actions):
            if action.name == ACTION_BUY:
                locations.append(ii)
        return np.array(locations)

    
    @cached_property
    def hold_lengths(self) -> np.ndarray:
        """Dates and time lengths of held stock.
        
        Returns
        -------
        buy_dates : np.ndarray[np.datetime64]
            Date where holding starts.
        sell_dates : np.ndarray[np.datetime64]
            Date where all stock is sold.
        lengths : np.ndarray[np.timedelta64]
            Lengths of time between buy and sell.

        """
        dates = [action.date for action in self.actions]
        holding_locs = self.hold_locations

        
        dates = np.concatenate(([dates[0]], dates, [dates[-1]]))
        holding_locs = np.concatenate(([False], holding_locs, [False]))
        holding_locs = holding_locs.astype(int)
        changes = np.diff(holding_locs)
        
        buy_locs = np.where(changes == 1)[0] + 1
        sell_locs = np.where(changes == -1)[0] + 1
        
        buy_dates = dates[buy_locs]
        sell_dates = dates[sell_locs]
        
        lengths = sell_dates - buy_dates
        return buy_dates, sell_dates, lengths
    
    
    @cached_property
    def _roi_calculations(self):
        """Calculations for return on investment -- net buys, net assets."""
        net_buys = 0
        net_sells = 0
        action : Action
        for action in self.actions:
            
            if action.name == ACTION_BUY:
                net_buys += action.amount
            
            elif action.name == ACTION_SELL:
                net_sells += action.gain
                
            elif action.name == ACTION_SELL_PERCENT:
                net_sells += action.gain
                
        assets = self.share_value + net_sells
        # out = (assets - net_buys)
        return net_buys, assets
    
    
    @cached_property
    def roi(self):
        """Net return on investment in dollar amount"""
        buys, assets = self._roi_calculations
        return assets - buys
    
    
    @cached_property
    def roi_ratio(self):
        """Return on investment as ratio of all purchases."""
        buys = self._roi_calculations[0]
        return self.roi / buys
    
    
    @cached_property
    def rate_of_return_yearly(self):
        """Yearly rate of return."""
        length = self.hold_lengths.sum()
        return self.roi_ratio * 365 / length
        
        
    @cached_property
    def share_value(self):
        """Current share value."""
        return self.transactions.get_share_valuation_for_action(-1)
    
    
class StockSnapshot:
    
    holdings_name = 'holdings ($)'
    stock_name = '__stock_name__'
    
    def __init__(self, strategy: Strategy, symbols=None):
        self._state = strategy.state
        self._strategy = strategy
        
        indicators = strategy.indicators().T
        stock_values = self._state.current_stocks
        
        if symbols is not None:
            indicators = indicators.loc[symbols]
            stock_values = stock_values[symbols]
        
        column = indicators.columns[0]
        self.indicators = indicators.sort_values(by=column, ascending=False)
        self.stock_values = stock_values
        self.symbols = self.stock_values.keys()
        
  
        
        
    def head(self, n: int=20):
        symbols = self.indicators.index.values[0 : n]
        return StockSnapshot(self._strategy, symbols=symbols)
        
    
    def tail(self, n: int=20):
        symbols = self.indicators.index.values[-n : ]
        return StockSnapshot(self._strategy, symbols=symbols)
    
    
    def sample(self, n: int=20):
        snum = len(self.symbols)
        n = min(n, snum)
        symbols = self.indicators.index.sample(n=n)
        return StockSnapshot(self._strategy, symbols=symbols)
    

        
    @cached_property
    def dataframe(self):
        df = self._strategy.indicators().T
        column = df.columns[0]
        df = df.sort_values(by=column, ascending=False)
        
        stock_values = self._state.current_stocks

        df[self.holdings_name] = stock_values
        df[self.holdings_name] = df[self.holdings_name].fillna(0)
        df[self.stock_name] = df.index
        return df
    
        
    def bar_bokeh(self):
        columns = self.indicators.columns
        df = self.dataframe
        
        plots = []
        for ii, name in enumerate(columns):
            plot = barplot(df=df, 
                           x=name,
                           y=self.stock_name,
                           c=self.holdings_name,  
                           tooltipcols=df.columns,
                           )
            plots.append(plot)
        return column(plots)
            
        
    def bar_pyplot(self):
        columns = self.dataframe.columns
        symbols = self.dataframe.index
        colnum = len(columns)

        f, axes = plt.subplots(
            colnum,
            1,
            figsize=(6.5, 1 * len(symbols)), 
            # sharex=True
            )
        try:
            axes[0]
        except TypeError:
            axes = [axes]
            
        for ii, colname in enumerate(columns):
            ax = axes[ii]
            sns.barplot(
                x=symbols, 
                y=self.dataframe[colname],
                hue=self.dataframe[self.holdings_name],
                ax=ax,
                palette='rocket',
                orient='h',
                )
            ax.axvline(0, color='k', linewidth=1.0)
            ax.set_xlabel(column)
            # pdb.set_trace()
            # sns.despine(bottom=True)
  
