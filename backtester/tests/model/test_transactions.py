# -*- coding: utf-8 -*-
import numpy as np
import pdb

from backtester.model import Action, Transactions
from backtester.definitions import ACTION_BUY, ACTION_SELL, ACTION_SELL_PERCENT
import datetime
from backtester.stockdata import YahooData
# a1 = Action()



def test_transations():
    GOOG = 'GOOG'
    MSFT = 'MSFT'
    
    date1 = datetime.datetime(2017, 1, 5)
    date2 = datetime.datetime(2017, 1, 20)
    date3 = datetime.datetime(2017, 5, 26)
    date4 = datetime.datetime(2018, 2, 26)
    
    a1 = Action(date=date1, symbol=GOOG, name=ACTION_BUY, amount=1000)
    a2 = Action(date=date2, symbol=GOOG, name=ACTION_SELL, amount=400)
    a3 = Action(date=date3, symbol=MSFT, name=ACTION_BUY, amount=700)
    a4 = Action(date=date4, symbol=GOOG, name=ACTION_SELL_PERCENT, amount=1)
    
    actions = [a1, a2, a3, a4]
    yahoo_data = YahooData()
    
    t1 = Transactions(
        stock_data = yahoo_data, 
        init_funds = 4000, 
        actions = actions
        )

    balance1 = t1.balances[0]
    balance2 = t1.balances[1]
    balance3 = t1.balances[2]
    balance4 = t1.balances[3]
    
    value1 = 1000
    assert balance1.available_funds == 4000 - value1
    assert balance1.equity == 4000
    
    value2 = a2.price * a2.shares
    value_sold = a2.price * (a1.shares - a2.shares)
    value_kept = a2.price * a1.shares - value_sold
    
    assert value_sold == 400
    assert balance2.equity == value2 + balance2.available_funds
    assert balance2.equity == value_kept + balance2.available_funds
    for aa in actions:
        print(aa)
    for bb in t1.balances:
        print(bb)
        
    df_a = t1.asset_history
    df_t = t1.dataframe  
    return

def test_interday():

    date1 = datetime.datetime(2017, 1, 5)
    date2 = datetime.datetime(2017, 1, 20, 0, 1)
    date3 = datetime.datetime(2017, 1, 20, 0, 2)
    date4 = datetime.datetime(2017, 1, 20, 0, 3)
    date5 = datetime.datetime(2017, 1, 20, 0, 4)
    date6 = datetime.datetime(2017, 1, 20, 0, 5)
    symbol1 = 'GOOG'
    symbol2 = 'MSFT'
    a1 = Action(date=date1, symbol=symbol1, name=ACTION_BUY, amount=1)
    a2 = Action(date=date2, symbol=symbol1, name=ACTION_SELL, amount=.5)
    a3 = Action(date=date3, symbol=symbol1, name=ACTION_SELL_PERCENT, amount=1)     
    a4 = Action(date=date4, symbol=symbol2, name=ACTION_BUY, amount=1)
    a5 = Action(date=date5, symbol=symbol2, name=ACTION_SELL, amount=1)
    a6 = Action(date=date6, symbol=symbol1, name=ACTION_BUY, amount=1)
    actions = [a1, a2, a3, a4, a5, a6]
    
    yahoo_data = YahooData()
    
    t1 = Transactions(
        stock_data = yahoo_data, 
        init_funds = 1, 
        actions = actions
        )
    
    df_a = t1.asset_history
    df_t = t1.dataframe
    
    v1 =  df_a['GOOG'] + df_a['MSFT'] + df_a['available_funds'] 
    v2 = df_a['equity']
    
    assert np.all(v1 == v2)
    
    return

    

def test_equity():
    yahoo = YahooData()
    dates = yahoo.get_trade_dates()
    date1 = np.datetime64('2016-01-01')
    date2 = np.datetime64('2016-02-01')
    
    t1 = Transactions(stock_data = yahoo, 
                      init_funds = 100,)
    
    dates = dates[(dates > date1) & (dates < date2)]
    symbol1 = 'SPY'
    
    for ii, date in enumerate(dates):
        
        if ii % 2 == 0:    
            t1.buy(date, symbol1, 1)
        else:
            t1.sell_percent(date, symbol1, 1)
            
    df1 = t1.dataframe
    del t1.dataframe
    t1._execute_equity()
    df2 = t1.dataframe
    
    # pdb.set_trace()
    
    assert np.all(df1.values == df2.values)
    return
        

def test_equity2():
    yahoo = YahooData()
    symbols = ['SPY', 'GOOG', 'TSLA', 'AAPL',]
    dates = yahoo.get_trade_dates()
    
    date1 = np.datetime64('2016-01-01')
    dates = dates[dates > date1][0:5]
    small = np.timedelta64(1, 'ms')
    
    transactions = Transactions(stock_data=yahoo, init_funds=100)
    for ii, date in enumerate(dates):
        # Buy
        if ii % 2 == 0:
            for symbol in symbols:
                transactions.buy(date, symbol, 10)
                date += small
        else:
            for symbol in symbols:
                transactions.sell_percent(date, symbol, 1)
                date += small
                
    df1 = transactions.dataframe
    # pdb.set_trace()
    del transactions.dataframe
    transactions._execute_equity()
    df2 = transactions.dataframe
    
    
    def check_column(name):
        try:
            assert np.all(np.isclose(df1[name], df1[name]))
        except TypeError:
            assert np.all(df1[name] == df2[name])
    for column in df1.columns:
        check_column(column)
    
    
def test_equity3():
    yahoo = YahooData()
    symbols = ['SPY', 'GOOG', 'TSLA', 'AAPL',]
    
    date = np.datetime64('2016-01-04')
    small = np.timedelta64(1, 'ms')

    transactions = Transactions(stock_data=yahoo, 
                                commission=0.0,
                                init_funds=100)
    for ii in range(10):
        if ii % 2 == 0:
            for symbol in symbols:
                transactions.buy(date, symbol, 10)
                date += small
        else:
            for symbol in symbols:
                transactions.sell_percent(date, symbol, 1)
                date += small
                
    df1 = transactions.dataframe
    assert np.all(df1['equity'] == 100)
    



if __name__ == '__main__':
    test_transations()
    test_interday()
    test_equity()
    test_equity2()
    test_equity3()
