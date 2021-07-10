# -*- coding: utf-8 -*-

from backtester.backtest import Strategy, Backtest
from backtester.model import TransactionsLastState, MarketState
from backtester.model import SymbolTransactions



class SymbolHistory:
    def __init__(self, symbol_transactions: SymbolTransactions):
        self.transactions = symbol_transactions
        self.actions = symbol_transactions.executed_actions
        
        
        
        
class SymbolsReport:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        
        
    def generate_step(self):
        """Build data for current step of strategy"""
        state = self.strategy.state
        return_ratios = state.return_ratios
        assets = state.current_stocks
        symbols = assets.columns
        