# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pytest
from backtester.reporting import SymbolReport
from backtester.model import SymbolTransactions, Action
from backtester.model import ACTION_BUY, ACTION_SELL_PERCENT, ACTION_HOLD
from backtester.stockdata import YahooData


class Scenario1:
    def __init__(self):
        yahoo = YahooData()
        
        date = np.datetime64('2016-01-04')
        date1 = date
        date2 = date + 4
        date3 = date + 8
        date4 = date + 15
        
        print(date1)
        print(date2)
        print(date3)
        print(date4)
        
        a1 = Action(date1, ACTION_BUY, amount=100, symbol='SPY')
        a2 = Action(date2, ACTION_SELL_PERCENT, amount=.5, symbol='SPY')
        a3 = Action(date3, ACTION_SELL_PERCENT, amount=1.0, symbol='SPY')
        a4 = Action(date4, ACTION_BUY, amount=50, symbol='SPY')
        actions = [a1, a2, a3, a4, ]    
        st = SymbolTransactions('SPY', yahoo)
        for action in actions:
            st.add(action)
        
        self.date1 = date1
        self.date2 = date2
        self.date3 = date3
        self.date4 = date4
        self.transactions = st
        self.actions = actions
        self.symbol_report = SymbolReport(st)
        
        
    
@pytest.fixture(scope='module')
def fixture1():
    return Scenario1()


def test_hold_lengths(fixture1):
    scenario1 = fixture1
    st = scenario1.transactions
    s = scenario1.symbol_report
    
    buy_dates, sell_dates, lengths = s.hold_lengths
    
    assert buy_dates[0] == scenario1.date1
    assert buy_dates[1] == scenario1.date4
    assert sell_dates[0] == scenario1.date3
    assert sell_dates[1] == scenario1.date4
    assert lengths[0] == 8
    assert lengths[1] == 0
    
    
def test_roi(fixture1):
    sc1 = fixture1
    s = sc1.symbol_report
    roi1 = s.roi
    
    date4 = sc1.date4
    sales = sc1.actions[1].gain + sc1.actions[2].gain
    buys = -sc1.actions[0].gain - sc1.actions[3].gain
    asset = sc1.transactions.get_share_valuation(date4)
    
    roi2 = (sales + asset - buys)
    assert roi1 == roi2 
    
    report = s
    pdb.set_trace()
    return


if __name__ == '__main__':
    test_hold_lengths(Scenario1())
    test_roi(Scenario1())
    
    
    