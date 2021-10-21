# -*- coding: utf-8 -*-

"""
Indicator bot
-------------
Indicator bot is a simple strategy that just picks stocks who have the 
highest indicator values on a given day.
"""
from backtester.backtest import Strategy



class IndicatorBot(Strategy):
    hold_period : int
    
    def init(self):
        self.current_stocks = set()
        self.ii = 0
        self.loss_days = 0
    
    
    def next(self):
        self.ii += 1
        if self.ii % self.hold_period == 0:
            stocks = self.get_stocks()
            self.buy_equally(stocks)
            
    
    def sell_all(self):
        if len(self.current_stocks) == 0:
            return
        
        print(self.date, 'Selling All')
        for stock in self.current_stocks:
            self.sell_percent(stock, amount=1.0)
        self.current_stocks = set()
    
        
    def buy_equally(self, symbols: set[str]):
        current_stocks = self.current_stocks.copy()
        if len(symbols) == 0:
            return
        
        if len(symbols) == len(current_stocks):
            if symbols == current_stocks:
                return
        
        
        assets = self.state.asset_values
        commission = self._transactions.commission
        
        equity = self.state.equity
        
        
        stocks_to_sell = current_stocks - symbols
        stocks_to_rebalance = symbols & current_stocks
        stocks_to_buy =  symbols - current_stocks
        
        curr_to_change = stocks_to_rebalance | stocks_to_sell
        curr_to_change = list(curr_to_change)
        # pdb.set_trace()
        fee = assets[list(stocks_to_sell)].sum() * commission
        # fee2 = assets[list(stocks_to_buy)].sum() * commission
        # fee = fee1 + fee2 
        
        # Stock asset value to set portfolio at.
        set_value = (equity - fee) / len(symbols)
        

        # Sell
        for ii, stock in enumerate(stocks_to_sell):
            self.sell_percent(stock, amount=1.0)

            
            

        # # Rebalance
        rebalance_delta = set_value - assets[stocks_to_rebalance]
        
        # Rebalance - Sell
        for stock in stocks_to_rebalance:
            delta = rebalance_delta[stock]
            if delta < 0:
                self.sell(stock, amount=-delta)
                
        # Rebalance - Buy
        
        stocks_to_rebuy = rebalance_delta[rebalance_delta > 0].index
        num_buy = len(stocks_to_buy) + len(stocks_to_rebuy)
        
        # Recalculate available funds or equity
        cash = self.state.available_funds
        rebuy_assets = assets[stocks_to_rebuy].sum()
        rebuy_equity = cash + rebuy_assets

        equity_per_buy = rebuy_equity / num_buy
        rebalance_buy = equity_per_buy - assets[stocks_to_rebalance]

        for stock in stocks_to_rebuy:
            delta = rebalance_buy[stock]
            self.buy(stock, amount=delta)       
                
        # Buy
        for stock in stocks_to_buy:
            # try:
                self.buy(stock, amount=equity_per_buy)
            # except:
            #     pdb.set_trace()
        print(self.date)
        print(self.state.current_stocks)
        self.current_stocks = symbols
        # pdb.set_trace()
        # print(self.state.available_funds)
        if self.state.available_funds > 1:
            raise ValueError('Something went wrong with this calculation.')
        return
    
    
    def get_stocks(self) -> set:
        MAX_ALLOWED = 30
        metrics = [self.stats.get(stock, self.date) for stock in STOCKS]
        metrics = np.array(metrics)
        isort = len(metrics) - np.argsort(metrics) - 1
        buy_indices1 = metrics > 0
        buy_indices2 = isort < MAX_ALLOWED
        buy_indices = buy_indices1 & buy_indices2
        
        new_stocks = STOCKS[buy_indices]
        # pdb.set_trace()
        return set(new_stocks)
    
    
    
    def get_metrics(self) -> pd.Series:
        metrics = [self.stats.get(stock, self.date) for stock in STOCKS]
        # metrics = np.array(metrics)
        metrics = pd.Series(metrics, index=STOCKS)
        metrics = metrics.sort_values(ascending=False)
        return metrics
    
    
    def get_metrics_dict(self):
        metrics = self.get_metrics()
        return pd.Series(metrics, index=STOCKS)
    