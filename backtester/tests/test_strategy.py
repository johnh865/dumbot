# -*- coding: utf-8 -*-
import pdb
import numpy as np

from backtester.backtest import Strategy, Backtest
from backtester.stockdata import YahooData, Indicators
from backtester.indicators import moving_avg
from backtester.exceptions import NoMoneyError
from datetime import datetime
from backtester.utils import crossover

yahoo = YahooData(['DIS', 'FNILX', 'GOOG', 'VOO'])

def test1():
    class MyStrategy(Strategy):
        def init(self):
            self.i1 = self.indicator(moving_avg, 25)
            
            
        def next(self):
            if self.days_since_start % 30 == 0:
                try:
                    print(f'Available cash {self.available_funds:.2f}')
                    b = self.buy('DIS', 250.0)
                    print(b)
                except NoMoneyError:
                    pass
                
            elif self.days_since_start % 100 == 0:
                s = self.sell_percent('DIS', 0.9)
                print(s)
            pass
    
    
    b = Backtest(stock_data=yahoo, 
                 strategy=MyStrategy,
                 cash=1000,
                 start_date = datetime(2018, 1, 1),
                 end_date = datetime(2019, 1, 1),
                 )
    b.start()
    df = b.stats.transactions
    assert len(df) > 0
    
    
def test2():
    """Test creating indicators."""
    class MyStrat(Strategy):
        def init(self):
            self.sma1 = self.indicator(moving_avg, 50)
            self.sma2 = self.indicator(moving_avg, 200)

        def next(self):
            # If sma1 crosses above sma2, close any existing
            # short trades, and buy the asset
            symbol = 'DIS'
            symbol2 = 'VOO'
            sma1 = self.sma1(symbol)
            sma2 = self.sma2(symbol)
                
            if crossover(sma1, sma2):
                t = self.sell_percent(symbol2, 1.0)
                print(t)
                t = self.buy(symbol, self.available_funds)
                print(t)

            # Else, if sma1 crosses below sma2, close any existing
            # long trades, and sell the asset
            elif crossover(sma2, sma1):
                t = self.sell_percent(symbol, 1.0)
                print(t)
                t = self.buy(symbol2, self.available_funds)
                print(t)
                
                
    bt = Backtest(stock_data=yahoo, 
                 strategy=MyStrat,
                 cash=1,
                 start_date = datetime(2014, 1, 1),
                 end_date = datetime(2019, 1, 1),
                 )
    bt.start()
    # df = b.transactions.stats.dataframe()
    # assert len(df) > 0
    # df2 = b.transactions.stats.asset_values()
    
    df1 = bt.stats.performance
    df2 = bt.stats.asset_values
    df3 = bt.transactions.dataframe
    return




def test_buy_hold():
    symbols = ['DIS', 'GOOG', 'VOO']
    
    # Define strategy
    class Holder(Strategy):
        def init(self):
            self.is_first = True
            
            
        def next(self):
            if self.is_first:
                self.is_first = False
                
                num = len(symbols)
                portion = self.available_funds / num
                for symbol in symbols:
                    self.buy(symbol, portion)
            return
    
    # Initialize test
    bt = Backtest(stock_data=yahoo, strategy=Holder,
                  cash=100, 
                  start_date=datetime(2014, 1, 1),
                  end_date=datetime(2019, 1, 1)
                  )
    bt.start()
    
    performance = bt.stats.performance
    assets = bt.stats.asset_values
    transactions = bt.stats.transactions        
        
    assert performance['DIS'][0] == 1
    assert performance['GOOG'][0] == 1
    assert performance['VOO'][0] == 1
    
    mystrat = performance['equity']
    stocks = performance[symbols].values
    stock_mean = stocks.mean(axis=1) 
    assert np.all(np.isclose(stock_mean, mystrat))
    
    return


if __name__ == '__main__':
    test1()
    test2()
    df = test_buy_hold()
    
    
    
    
    
    
    