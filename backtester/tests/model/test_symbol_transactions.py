# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pytest 

from backtester.model import Action, SymbolTransactions
from backtester.definitions import ACTION_BUY, ACTION_SELL, ACTION_SELL_PERCENT
import datetime
from backtester.stockdata import YahooData
from backtester.exceptions import BacktestError
# a1 = Action()


yahoo_data = YahooData()

def test_last_trading_date():
    sh = SymbolTransactions('GOOG', yahoo_data, commission=0.0)
    date = datetime.datetime(2017, 2, 25)
    date = np.datetime64(date)
    
    last_date = sh.get_previous_trading_date(date)
    next_date = sh.get_next_trading_date(date)
    
    print('date: ', date)
    print('last trade date: ', last_date)
    print('next trade date: ', next_date)
    
    dates = sh.df.index
    assert last_date <= date
    assert next_date >= date
    return


def test_last_trading_date2():
    sh = SymbolTransactions('GOOG', yahoo_data, commission=0.0)
    date = datetime.datetime(2017, 2, 24)
    
    last_date = sh.get_previous_trading_date(date)
    next_date = sh.get_next_trading_date(date)
    
    print('date: ', date)
    print('last trade date: ', last_date)
    print('next trade date: ', next_date)
    
    dates = sh.df.index
    assert last_date < np.datetime64(date)
    assert next_date == np.datetime64(date)
    return



def test_3():
    symbol = 'GOOG'
    date1 = datetime.datetime(2017, 1, 5)
    date2 = datetime.datetime(2017, 1, 20)
    date3 = datetime.datetime(2017, 5, 25)
    date1 = np.datetime64(date1)
    date2 = np.datetime64(date2)
    date3 = np.datetime64(date3)
    
    
    
    a1 = Action(date=date1, symbol=symbol, name=ACTION_BUY, amount=1000)
    a2 = Action(date=date2, symbol=symbol, name=ACTION_SELL, amount=400)
    a3 = Action(date=date3, symbol=symbol, name=ACTION_SELL_PERCENT, amount=1)
    actions = [a1, a2, a3]
    sh = SymbolTransactions(symbol=symbol,
                            stock_data=yahoo_data,
                            actions=actions)
    
    # Test BUY
    assert a1.amount == 1000
    assert a1.gain == -a1.amount
    assert a1.shares == a1.amount / a1.price 
    
    # Test SELL
    assert a2.amount == 400
    assert a2.gain == 400
    assert a2.shares == a1.shares - a2.amount / a2.price
    
    # TEST SELL_ALL
    assert a3.shares == 0 
    assert a3.gain == a2.shares * a3.price
    
    # assert sh.gain == (a1.gain + a2.gain + a3.gain)
    
    
    # Test share value
    shares1 = sh.get_shares(date1)
    shares2 = sh.get_shares(date2)
    shares3 = sh.get_shares(date3)
    assert shares1 == a1.shares
    assert shares2 == a2.shares
    assert shares3 == a3.shares
    
    date4 = datetime.datetime(2018, 1, 1)
    date4 = np.datetime64(date4)
    
    shares4 = sh.get_shares(date4)
    assert shares4 == a3.shares
    return


    

def test_exec_one():
    symbol = 'GOOG'
    date1 = datetime.datetime(2017, 1, 5)
    date2 = datetime.datetime(2017, 1, 20)
    date3 = datetime.datetime(2017, 5, 25)
    
    a1 = Action(date=date1, symbol=symbol, name=ACTION_BUY, amount=1000)
    a2 = Action(date=date2, symbol=symbol, name=ACTION_SELL, amount=400)
    a3 = Action(date=date3, symbol=symbol, name=ACTION_SELL_PERCENT, amount=1)  
    actions = [a1, a2, a3]
    sh = SymbolTransactions(
        symbol=symbol,
        stock_data=yahoo_data,
        actions=actions)
    sh.execute()
    
    
    b1 = Action(date=date1, symbol=symbol, name=ACTION_BUY, amount=1000)
    b2 = Action(date=date2, symbol=symbol, name=ACTION_SELL, amount=400)
    b3 = Action(date=date3, symbol=symbol, name=ACTION_SELL_PERCENT, amount=1)  
    actions2 = [b1, b2, b3]
    sh = SymbolTransactions(symbol=symbol, stock_data=yahoo_data)
    sh.add(b1)
    sh.add(b2)
    sh.add(b3)
    
    for a, b, in zip(actions, actions2):
        
        assert a.shares == b.shares
        assert a.date == b.date
        assert a.amount == b.amount
        assert a.gain == b.gain
        assert a.price == b.price
        print(a)
        print(b)
        

        
def test_interday():
    date1 = datetime.datetime(2017, 1, 5)
    date2 = datetime.datetime(2017, 1, 20, 0, 1)
    date3 = datetime.datetime(2017, 1, 20, 0, 2)
    date4 = datetime.datetime(2017, 1, 20, 0, 3)
    symbol = 'GOOG'
    
    a1 = Action(date=date1, symbol=symbol, name=ACTION_BUY, amount=1)
    a2 = Action(date=date2, symbol=symbol, name=ACTION_SELL, amount=.5)
    a3 = Action(date=date3, symbol=symbol, name=ACTION_SELL_PERCENT, amount=1)     
    a4 = Action(date=date4, symbol=symbol, name=ACTION_BUY, amount=1) 
    
    actions = [a1, a2, a3, a4]
    sh = SymbolTransactions(symbol=symbol, 
                            actions=actions,
                            stock_data=yahoo_data)
    sh.execute()
    
    action: Action
    for action in actions:
        date = action.date
        print(action)
        
            
def test_return_ratio():    
    date1 = np.datetime64('2017-01-05')
    date2 = np.datetime64('2017-01-20')
    date3 = np.datetime64('2017-05-24')
    date4 = np.datetime64('2017-08-25')
    date5 = np.datetime64('2017-09-25')
    
    
    symbol = 'GOOG'

    a1 = Action(date=date1, symbol=symbol, name=ACTION_BUY, amount=1000)
    a2 = Action(date=date2, symbol=symbol, name=ACTION_SELL, amount=400)
    a3 = Action(date=date3, symbol=symbol, name=ACTION_SELL_PERCENT, amount=1)  
    a4 = Action(date=date4, symbol=symbol, name=ACTION_BUY, amount=100)  
    
    actions = [a1, a2, a3, a4,]
    
    sh = SymbolTransactions(symbol=symbol, stock_data=yahoo_data)
    sh.add(a1)

    ratio = sh.last_return_ratio(date2)
    ratio_r = (sh.get_price(date2) - sh.get_price(date1)) / sh.get_price(date1)
    assert np.isclose(ratio, ratio_r)
    
    sh.add(a2)
    sh.add(a3)
    ratio = sh.last_return_ratio(date3)
    assert ratio == 0
    
    ratio = sh.last_return_ratio(date4)
    assert ratio == 0
    
    sh.add(a4)
    ratio = sh.last_return_ratio(date4)
    assert ratio == 0
    
    ratio = sh.last_return_ratio(date5)
    ratio_r = (sh.get_price(date5) - sh.get_price(date4)) / sh.get_price(date4)
    assert np.isclose(ratio, ratio_r)
    
    # Test to make sure error is raised when previous date is used. 
    with pytest.raises(BacktestError):
        sh.last_return_ratio(date3)
        
        
    
    
    return

    
    
def test_share_valuation():
    symbol = 'SPY'
    yahoo = YahooData()
    
    date = np.datetime64('2016-01-04')
    small = np.timedelta64(1, 'ms')
    
    
    st = SymbolTransactions(symbol, yahoo)
    dates = []
    for ii in range(10):
        dates.append(date)
        if ii % 2 == 0:
            action = Action(date, ACTION_BUY, amount=10, symbol=symbol)
        else:
            action = Action(date, ACTION_SELL, amount=10, symbol=symbol)
        st.add(action)
        date += small
        
        
    dates = np.array(dates)
    values1 = st.get_share_valuations(dates)
    values2 = [st.get_share_valuation_for_action(ii) for ii in range(10)]
    values2 = np.array(values2)
    assert np.all(values1 == values2)
    
    
if __name__ == '__main__':
    test_3()
    test_last_trading_date()
    test_last_trading_date2()
    test_exec_one()
    test_interday()
    test_share_valuation()
    test_return_ratio()
