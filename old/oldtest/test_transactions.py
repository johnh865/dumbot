# -*- coding: utf-8 -*-


from dumbot.dataclasses import Action, SymbolTransactions, Transactions
from dumbot.dataclasses import ACTION_BUY, ACTION_SELL, ACTION_SELL_PERCENT
import datetime

# a1 = Action()



def test_transations():
    GOOG = 'GOOG'
    MSFT = 'MSFT'
    
    date1 = datetime.datetime(2017, 1, 5)
    date2 = datetime.datetime(2017, 1, 20)
    date3 = datetime.datetime(2017, 5, 25)
    date4 = datetime.datetime(2018, 2, 25)
    
    a1 = Action(date=date1, symbol=GOOG, name=ACTION_BUY, amount=1000)
    a2 = Action(date=date2, symbol=GOOG, name=ACTION_SELL, amount=400)
    a3 = Action(date=date3, symbol=MSFT, name=ACTION_BUY, amount=700)
    a4 = Action(date=date4, symbol=GOOG, name=ACTION_SELL_PERCENT, amount=1)
    
    actions = [a1, a2, a3, a4]
    
    t1 = Transactions(init_funds = 4000, actions=actions)

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
    return

        
if __name__ == '__main__':
    test_transations()
