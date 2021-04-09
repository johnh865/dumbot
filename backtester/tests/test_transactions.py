# -*- coding: utf-8 -*-
import numpy as np

from dumbot.model import Action, Transactions
from dumbot.definitions import ACTION_BUY, ACTION_SELL, ACTION_SELL_PERCENT
import datetime
from dumbot.stockdata import YahooData
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
        
    df_a = t1.asset_history()
    df_t = t1.dataframe()    
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
    
    df_a = t1.asset_history()
    df_t = t1.dataframe()
    
    v1 =  df_a['GOOG'] + df_a['MSFT'] + df_a['available_funds'] 
    v2 = df_a['equity']
    
    assert np.all(v1 == v2)
    
    return

    





if __name__ == '__main__':
    test_transations()
    test_interday()
