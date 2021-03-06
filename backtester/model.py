import pdb
import datetime
import dataclasses
from collections import deque 
from abc import abstractmethod, ABCMeta


from warnings import warn
from functools import cached_property

import numpy as np
import pandas as pd



from sqlalchemy import create_engine

from backtester.definitions import (CONNECTION_PATH, DF_ADJ_CLOSE, DF_DATE,
                                DF_HIGH, DF_LOW, DF_CLOSE)
from backtester.definitions import (ACTION_BUY, ACTION_SELL, ACTION_DEPOSIT, 
                                ACTION_WITHDRAW, ACTION_HOLD,
                                ACTION_SELL_PERCENT)
from backtester.definitions import SMALL_DOLLARS

from datasets.symbols import ALL
from backtester.stockdata import BaseData
from backtester.utils import dates2days, floor_to_date
from backtester.utils import InterpConstAfter, delete_attr
from backtester.exceptions import NoMoneyError, TradingError, BacktestError
from backtester import utils
from backtester.exceptions import DataError


@dataclasses.dataclass
class Action:
    """State immediately **after** performance of an action. 
    
    Parameters
    ----------
    date : datetime.datetime
        Date transaction executed.
    symbol : str or None
        stock market symbol, None for deposit/withdraw actions. 
    name : str
        Transaction type.
    amount : float
        Total dollar amount of transaction including fees. 
    fee : float
        Fees, commissions, etc of transaction.
        
    shares : float
        Number of shares after end of transaction
    gain : float
        Increase or decrease of amount at end of transaction
    price : float
        Price of stock symbol.
    
    """
    date : np.datetime64
    name : str
    amount : float
    symbol : str = None

    shares : float = None
    fee : float = 0.0
    gain : float = 0.0
    price : float = None
    
    def __post_init__(self):
        self.date = np.datetime64(self.date)


    def __str__(self):
        date = self.date
        name = self.name
        symbol = self.symbol
        dstr = str(date)
        # dstr = str(date).split('T')[0]
        
        base = f'Action(date={dstr}, name={name}, symbol={symbol}'
        if self.amount is not None:
            if self.amount > 0:
                base += f', amount={self.amount:.2f}'
        if self.price is not None:
            base += f', price={self.price:.2f}'
        if self.gain > 0:
            base += f', gain={self.gain:.2f}'
        base += ')'
        return base
    
    def __repr__(self):
        return self.__str__()
    
    
    def to_series(self):
        d = dataclasses.asdict(self)
        return pd.Series(d)

    


@dataclasses.dataclass
class AccountBalance:
    """
    Parameters
    ----------
    date : datetime.datetime
        Date balance is calculated.
    available_funds : float
        Cash on hand
    equity : float
        Cash + assets value
    """
    date : np.datetime64
    available_funds : float = None
    equity : float = None
    

    def __str__(self):
        date = self.date
        dstr = str(date).split('T')[0]
        
        base = f'AccountBalance(date={dstr}'
        base += f', available_funds={self.available_funds:.2f}'
        base += f', equity={self.equity:.2f}'
        base += ')'
        return base
    
    
    def __post_init__(self):
        self.date = np.datetime64(self.date)

    
    
    def __repr__(self):
        return self.__str__()
    
    
    def to_series(self):
        d = dataclasses.asdict(self)
        return pd.Series(d)
    
    
    
class TransactionsDF(pd.DataFrame):
    """Explicit definition of columns found in Transactions dataframe."""
    available_funds : pd.Series = None
    equity : pd.Series = None
    date : pd.Series = None
    
    name : str = None
    amount : float = None
    symbol : str = None

    shares : float  = None
    fee : float = None
    gain : float  = None
    price : float  = None
    
    
class ActionExecutor(metaclass=ABCMeta):
    def __init__(self, transactions: 'SymbolTransactions'):
        self.transactions = transactions
        self._executed_num = 0
    
    
    def execute(self):
        actions = self.transactions.executed_actions
        while self._executed_num < len(actions):
            action = actions[self._executed_num]
            self._execute_one(action)
            self._executed_num += 1
            
            
    @abstractmethod
    def execute_one(self, action: Action):
        return
    
            
            
        
        
# class SymbolCostBasis(ActionExecutor):
#     """Calculate stock cost basis, the money you've spent on purchases."""
#     def __init__(self, transactions: SymbolTransactions):
#         super().__init__(transactions)
#         self.costs = []
#         self.equity = []
#         self.


        
#     def execute_one(self, action: Action):
#         last_cost = self.costs[-1]
#         if action.name == ACTION_BUY:
#             new_cost = last_cost + action.amount
            
        
        
        
        
        
#     def get(self, date: np.datetime64):
        
        
    
    
    
    

class SymbolTransactions:
    """Store and process trades for a particular symbol."""
    def __init__(self, 
                 symbol: str, 
                 stock_data: BaseData,
                 actions: list[Action]=(), 
                 commission=0.0,
                 price_name: str=DF_ADJ_CLOSE
                 ):
        """        
        Parameters
        ----------
        symbol : str
            Stock ticker symbol.
        stock_data : BaseData
            StockData.
        actions : list[Action], optional
            Initial trade actions. The default is [].
        commission : TYPE, optional
            Broker commission per trade. The default is 0.0.

        """
        
        self.price_name = price_name
        self.symbol = symbol
        self.commission = commission
        self.queue = deque(actions)
        self.executed_actions = []
        # self.df = read_dataframe(symbol)
        
        self.df = stock_data.dataframes[symbol]
        self.execute()
        
        
    # def _get_date_loc(self, date):
    #     date = np.datetime64(date)
    #     try:
    #         iloc = np.where(self.df.index == date)[0]
    #     except IndexError:
    #         raise ValueError(
    #             f'Date {date} was not found, probably not a trading day.'
    #             )
    #     return iloc
    
    def get_previous_trading_date(self, date: np.ndarray):
        """Get previous trading date from current `date`."""
        dates0 = utils.floor_dates(date)        
        dates = self.df.index 
        ii = np.searchsorted(dates, dates0)
        return dates[ii - 1]
    

    
    def get_next_trading_date(self, dates: np.ndarray) -> np.ndarray:
        """Get next available trading date. 
        Array values are np.nan if particular date not available. """
        
        dates0 = utils.floor_dates(dates)
        dates1 = self.df.index.values
        dlen = len(dates1)
        ii = np.searchsorted(dates1, dates0)
        
        ii_no_data = ii >= dlen
        # jj = ii[ii_no_data]
        
        try:
            ii[ii_no_data] = dlen - 1
        except TypeError:
            if ii_no_data:
                ii = dlen - 1
            
            
        out = dates1[ii]
        # out[jj] = np.nan
        return out
    
        
        # try:
        #     return dates1[ii]
        # except IndexError:
        #     symbol = self.symbol
        #     s = f'Data "{self.price_name}" not found for {dates} for symbol {symbol}.'
        #     raise DataError(s)
        
    
    # def _get_previous_close(self, date):
    #     date = np.datetime64(date)
        
    #     try:
    #         out = self.df.loc[date]
    #     except KeyError:
    #         warn(
    #             f'Date {date} was not found, probably not a trading day.'
    #             )
    #         date = self.get_previous_trading_date(date)
    #         return self._get_previous_close(date)
        
    #     return out[DF_ADJ_CLOSE], date
    
    # def _get_close(self, date):
    #     date = np.datetime64(date)
    #     return self.df.loc[date]
    
    
    def get_next_close(self, date: np.ndarray) -> np.ndarray:
        """Get next close (adjusted) price given `date`."""
        next_dates = self.get_next_trading_date(date)
        return self._price_interpolator(next_dates)
        # dates = self.df.index
        # values = self.df[self.price_name]   
        
        # return 
        # return interp_const_after(dates, values, next_dates)
    
    
    @cached_property
    def _price_interpolator(self):
        dates = self.df.index.values
        values = self.df[self.price_name].values
        return InterpConstAfter(dates, values)

    
    @cached_property
    def _share_interpolator(self):
        df = self.dataframe
        shares = df['shares'].values
        times = df.index.values
        interp = InterpConstAfter(times, shares, before=0)
        return interp
    
    def get_share(self, date: np.datetime64) -> float:
        return self._share_interpolator.scalar(date)
    
    
    def get_shares(self, dates: np.ndarray) -> float:
        return self._share_interpolator(dates)
    
    
    
    def last_return_ratio(self, date: np.datetime64) -> float:
        """Calculate ratio of return price since last buy from zero shares."""
        df = self.dataframe
        shares = df['shares'].values
        prices = df['price'].values
        
        if len(prices) == 0:
            return 0.0
        if shares[-1] == 0:
            return 0.0
        
        try:
            start_loc = np.where(shares == 0)[0][-1] + 1
            
            if df.index[start_loc] > date:
                raise BacktestError('Date must be greater than date of last action.')
        except IndexError:
            start_loc = 0
        
        start_price = prices[start_loc]
            
        end_price = self.get_price(date) * (1 - self.commission)
        ratio = (end_price - start_price) / start_price
        return ratio
   
        

        
        
    # def _get_next_close(self, date):
    #     """Get next close (adjusted) price given `date`."""
    #     date = np.datetime64(date)
    #     date = floor_to_date(date)
    #     try:
    #         out = self.df.loc[date]
    #     except KeyError:
    #         date1 = self.get_next_trading_date(date)
            
    #         warn(
    #             f'Date {date} was not found, probably not a trading day.'
    #             f' New date is {date1}.'
    #             )
    #         return self._get_next_close(date1)
        
    #     return out[self.price_name], date    
    
    
    # def get_shares(self, dates: np.ndarray) -> np.ndarray:
    #     """Get the number of invested shares for given dates."""
    #     df = self.dataframe
    #     shares = df['shares'].values
    #     times = df.index.values
    #     dates = utils.datetime_to_np(dates)
    #     # dates = np.asarray(dates).astype('datetime64')
    #     out = interp_const_after(times, shares, dates, before=0.0)
    #     return out
    
    
    def last_shares(self):
        """Get shares for last executed action"""
        action = self.executed_actions[-1]
        return action.shares
    
    
    
    def get_share_valuation_for_action(self, ii:int):
        """Retrieve share value for a specific action index."""
        action = self.executed_actions[ii]
        shares =  action.shares
        if shares == 0:
            return 0.0
        
        date = action.date
        date = floor_to_date(date)
        try:
            close = self.df[self.price_name][date]
        except KeyError:
            close = self.get_next_close(date)
        return close * shares
        
        
    
    # def _get_shares(self, date: datetime.datetime) -> float:
    #     """Get the number of invested shares for a given date."""
    #     # dates = [action.date for action in self.actions]
    #     actions = self.executed_actions
        
    #     if date < actions[0].date:
    #         return 0.0
    #     if len(actions) == 1:
    #         return actions[0].shares
        
    #     for ii in range(len(actions)-1):
    #         action1 = actions[ii]
    #         action2 = actions[ii + 1]
    #         if action1.date <= date < action2.date:
    #             return action1.shares
            
        
    #     # If date has gone through whole history, return last share value.
    #     return action2.shares
    
    
    def get_share_valuations(self, date: np.ndarray) -> np.ndarray:
        """Calculate value of shares at the given dates."""
        price = self.get_prices(date)
        shares = self.get_shares(date)
        value = price * shares
        return value
    
    
    def get_share_valuation(self, date: np.datetime64) -> float:
        """Calculate value of shares at the given date."""
        price = self._price_interpolator.scalar(date)
        shares = self._share_interpolator.scalar(date)
        value = price * shares
        return value    
    
    
    def get_price(self, date: np.datetime64) -> float:
        return self._price_interpolator.scalar(date)
    
    
    def get_prices(self, dates: np.ndarray) -> np.ndarray:
        return self._price_interpolator.array(dates)
    

    
    # def get_adj_price(self, date: np.ndarray) -> np.ndarray:
    #     """Estimate the price of the stock if bought in the date."""
    #     close = self._interp_close
    #     times = self._interp_times
    #     date = date.astype(float)
    #     return self._price_interpolator(date) 
    
    
    # @cached_property
    # def _interp_close(self):
    #     return self.df[self.price_name].values
    
    
    # @cached_property
    # def _interp_times(self):
    #     return self.df.index.values.astype(float)
    
    
    
    
    def _check_dates(self, action: Action):
        """Error check to make sure date is allowed."""
        date = action.date
        try:
            last_date= self.executed_actions[-1].date
        except IndexError:
            return
        
        if date <= last_date:
            s = (f'Transaction at {date} is not allowed on equal or before'
                 f' the last action at {last_date}.')
            raise TradingError(s)
            
        
        # Check to make sure date is in the data index. 
        date1 = np.datetime64(date, 'D')  
        dates = self.df.index.values
        loc = np.searchsorted(dates, date1)
        
        if dates[loc] != date1:
            raise TradingError(f'{date} is not a valid trading day.')
        
    
    def _execute_one(self):
        action = self.queue.popleft()
        
        try:
            last_action = self.executed_actions[-1]
        except IndexError:
            shares = 0
        else:
            shares = last_action.shares
        
        # Check that there are no duplicate dates
        self._check_dates(action)
        
        date = action.date
        # trade_date = self.get_next_trading_date(date)
        price = self.get_price(date)

        # close, date = self._get_last_close(date)
        amount = action.amount
        name = action.name    
    
        # Update action information
        action.price = price
        # action.date = trade_date    
        
     
        if name == ACTION_BUY:
            fee = amount * self.commission
            shares += (amount - fee) / price
            action.shares = shares
            action.gain = -amount
            action.fee = fee
            
        elif name == ACTION_SELL:
            fee = amount * self.commission
            shares_sold = (amount - fee) / price
            shares += -shares_sold
            action.shares = shares
            action.gain = amount - fee
            action.fee = fee
            
            if shares < 0:
                raise TradingError(
                    f'{shares_sold: .2f} amount of shares sold exceeds'
                    f'{action.shares: .2f} amount owned.')
            
        elif name == ACTION_SELL_PERCENT:
            fraction = amount
            shares_sold = shares * fraction
            shares_kept = shares * (1 - fraction)
            sale_amount = shares_sold * price
            fee = sale_amount * self.commission
            
            action.shares = shares_kept
            action.gain = sale_amount - fee
            action.fee = fee
            
        self.executed_actions.append(action)
        
        return
    
    
    def execute(self):
        delete_attr(self, 'dataframe')
        delete_attr(self, '_price_interpolator')
        delete_attr(self, '_share_interpolator')
        while self.queue:
            self._execute_one()            

    
    def add(self, action):
        self.queue.append(action)
        return self.execute()
    
    
    @cached_property
    def dataframe(self) -> pd.DataFrame:
        """Save actions and balances to dataframe. See `Action` for columns. 
        
        Returns
        -------
        out : pandas.DataFrame
            - index -- dates associated with actions.        
        """
    
        data1 = [b.to_series() for b in self.executed_actions]
        df1 = pd.DataFrame(data1)
        df1 = df1.set_index('date')
        return df1
    
        


class Transactions:
    """Store and process stock transactions."""
    def __init__(self, 
                 stock_data: BaseData, 
                 init_funds=0, 
                 actions: list[Action]=None, 
                 commission=0.0,
                 price_name=DF_ADJ_CLOSE,
                 ):
        self.stock_data = stock_data
        self.init_funds = init_funds
        self.commission = commission
        self.executed_actions = []
        self.balances = []
        self.price_name = price_name
        
        # Keep track of active stocks for latest action
        self.latest_portfolio = set()
        
        # self.stats = TransactionStats(self)
        
        if actions is None:
            self.queue = deque()
        else:
            self.queue = deque(actions)
            self.execute()

        
        return
    
    
    def reset(self):
        """Clear transaction data."""
        self.executed_actions = []
        self.balances = []
        self.queue = deque()
        delete_attr(self, '_symbol_transactions_dict')
        delete_attr(self, 'dataframe')
        delete_attr(self, 'asset_history')
    
    
    def _get_symbol_transactions_dict(self) -> dict[str, SymbolTransactions]:
        try:
            sdict = getattr(self, '_symbol_transactions_dict')
            return sdict

        except AttributeError:
            self._symbol_transactions_dict = {}
            return self._symbol_transactions_dict
        
        
    def get_symbol_transactions(self, symbol:str) -> SymbolTransactions:
        return self._get_symbol_transactions_dict()[symbol]
    
    
    def get_symbols_list(self) -> list[str]:
        return list(self._get_symbol_transactions_dict().keys())
        
    
    
    def _execute_symbol(self, action: Action):
        
        # Retrieve Symbol Transactions storage dict
        sdict = self._get_symbol_transactions_dict()
                
        name = action.name
        symbol = action.symbol
        
        if (name == ACTION_BUY or 
            name == ACTION_SELL or
            name == ACTION_SELL_PERCENT):
            if symbol in sdict:
                st = sdict[symbol]
            else:
                st = SymbolTransactions(
                    symbol, 
                    stock_data=self.stock_data,
                    commission=self.commission,
                    price_name=self.price_name,
                    )
                sdict[symbol] = st
                
            # Execute using add.
            st.add(action)
        return  
    
    
    def _check_action_timestamp(self, action: Action):
        """Make sure new action is newer than previous actions."""
        try:
            last_action = self.executed_actions[-1]
        except IndexError:
            return
        
        last_date = last_action.date
        if action.date <= last_date:
            s = (f'Date for {action} '
                 f'must be newer than last action at {last_date}.')
            raise TradingError(s)
                    
            
    def _execute_one(self):
        action = self.queue.popleft()
        name = action.name
        date = action.date
        
        # Check to make sure action has unique timestamp.
        self._check_action_timestamp(action)

        try:
            last_balance: AccountBalance = self.balances[-1]
        except IndexError:
            available_funds = self.init_funds
        else:
            available_funds = last_balance.available_funds

        if name == ACTION_BUY:
            if available_funds - action.amount + SMALL_DOLLARS < 0:
                action.amount = 0 
                raise NoMoneyError(f'Available funds is less than 0 from {action}!')

        self._execute_symbol(action)


        if name == ACTION_DEPOSIT:
            available_funds += action.amount
            
        elif name == ACTION_WITHDRAW:
            available_funds += -action.amount
            
        elif name == ACTION_BUY:
            available_funds += -action.amount
            self.latest_portfolio.add(action.symbol)
            
        elif name == ACTION_SELL:
            available_funds += action.gain
            if action.shares == 0:
                self.latest_portfolio.remove(action.symbol)
            
        elif name == ACTION_SELL_PERCENT:
            available_funds += action.gain          
            if action.shares == 0:
                self.latest_portfolio.remove(action.symbol)
        
        elif name == ACTION_HOLD:
            pass
                    
        balance = AccountBalance(
            date = date, 
            available_funds = available_funds,
            equity = None)
            
        self.balances.append(balance)
        self.executed_actions.append(action)
        self._execute_last_equity()
        return
    
        
    def execute(self):
        delete_attr(self, 'dataframe')
        delete_attr(self, 'asset_history')
        
        while self.queue:
            self._execute_one()            

    
    def add(self, action):
        self.queue.append(action)
        self.execute()
        return action
    
    
    def buy(self, date: datetime.datetime, symbol: str, amount: float):
        action = Action(date, name=ACTION_BUY, symbol=symbol, amount=amount)
        return self.add(action)
    
    
    def sell(self, date: datetime.datetime, symbol: str, amount: float):
        action = Action(date, name=ACTION_SELL, symbol=symbol, amount=amount)
        return self.add(action)
    
    
    def sell_percent(self, date: datetime.datetime, symbol: str, amount: float):
        action = Action(date,
                        name=ACTION_SELL_PERCENT, 
                        symbol=symbol, amount=amount)
        return self.add(action)
    
    
    def hold(self, date: datetime.datetime):
        action = Action(date, name=ACTION_HOLD, amount=0.0)
        return self.add(action)
    
    
    def _execute_last_equity(self):
        """Calculate equity (cash + assets) for the last transaction."""
        balance : AccountBalance
        balance = self.balances[-1]
        
        equity = balance.available_funds
        
        sdict = self._get_symbol_transactions_dict()
        portfolio = self.latest_portfolio
        
        for symbol in portfolio:
            s_transaction = sdict[symbol]
            value = s_transaction.get_share_valuation(balance.date)
            equity += value
        
        balance.equity = equity
        return
        
        
    def _execute_equity(self):
        """Re-Calculate equity (cash + assets)"""
        sdict = self._get_symbol_transactions_dict()
        actions = self.executed_actions
        balances = self.balances

        dates = np.array([a.date for a in actions])
        equity = np.array([b.available_funds for b in balances])
        
        for s_transaction in sdict.values():
            values = s_transaction.get_share_valuations(dates)
            equity += values

        for ii, balance in enumerate(balances):
            balance.equity = equity[ii]
        return
    
    
    # def last_equity(self):
    #     try:
    #         return self.balances[-1].net_assets
    #     except IndexError:
    #         return self.init_funds
        
        
    def get_asset_values(self, date: np.datetime64) -> pd.Series:
        """Retrieve value of each assets."""
        sdict = self._get_symbol_transactions_dict()
        new = {}
        st : SymbolTransactions
        for symbol, st in sdict.items():
            date = np.datetime64(date)
            value = st.get_share_valuation(date)
            new[symbol] = value
        return pd.Series(new)
    
    
    def get_asset_shares(self, date: np.datetime64) -> pd.Series:
        """Retrieve # of shares for each asset."""
        sdict = self._get_symbol_transactions_dict()
        new = {}
        st : SymbolTransactions
        for symbol, st in sdict.items():
            date = np.datetime64(date)
            value = st.get_share(date)
            new[symbol] = value
        return pd.Series(new)    
    
    
    def get_traded_symbols(self):
        """Retrun list[str] of traded symbols."""
        return list(self._get_symbol_transactions_dict().keys())
    
    
    
    def get_return_ratios(self, date: np.datetime64):
        sdict = self._get_symbol_transactions_dict()
        new = {}
        st : SymbolTransactions
        for symbol, st in sdict.items():
            date = np.datetime64(date)
            value = st.last_return_ratio(date)
            new[symbol] = value
        return pd.Series(new)    
        
        
        
    def get_available_funds(self, date: np.datetime64):
        """Get available cash funds for a given date."""
        balances = self.balances
        balance1 : AccountBalance
        balance2 : AccountBalance
        
        if len(self.balances) == 0:
            return self.init_funds 
     
        if date < balances[0].date:
            return self.init_funds 
        
        if len(balances) == 1:
            return balances[0].available_funds
        
        for ii in range(len(balances)-1):
            balance1 = balances[ii]
            balance2 = balances[ii + 1]
            if balance1.date <= date < balance2.date:
                return balance1.available_funds
        
        # If date has gone through whole history, return last share value.
        return balance2.available_funds        
        
        
    def last_available_funds(self):
        """Retrieve funds from the last transaction."""
        try:
            return self.balances[-1].available_funds
        except IndexError:
            return self.init_funds
        
        
        
    @cached_property
    def dataframe(self) -> pd.DataFrame:
        """Save actions and balances to dataframe."""
    
        data1 = [b.to_series() for b in self.balances]
        data2 = [b.to_series() for b in self.executed_actions]
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        df1 = df1.set_index('date')
        df2 = df2.set_index('date')
        df1[df2.columns] = df2
        return df1


    @cached_property
    def asset_history(self) -> pd.DataFrame:
        """Save asset values for all action dates."""
        
        balances = self.balances
        
        dates = [b.date for b in balances]
        cash = [b.available_funds for b in balances]
        equity = [b.equity for b in balances]
        
        datas = []
        for b in balances:
            date = b.date
            v = self.get_asset_values(date)
            datas.append(v)
            
        df = pd.DataFrame(datas)
        df[DF_DATE] = dates
        df['available_funds'] = cash
        df['equity'] = equity
        
        df = df.set_index(DF_DATE, drop=True)
        return df
    


class TransactionsLastState:
    """Get state for a date after the last transaction."""
    def __init__(self, transactions: Transactions, date: np.datetime64):
        self._transactions = transactions
        self._stock_data = transactions.stock_data
        self.date = date
        
        actions = self._transactions.executed_actions
        
        if len(actions) > 0:
            last_action = actions[-1]
            last_transaction_date = last_action.date
            if last_transaction_date > date:
                raise BacktestError(
                    f'date {date} cannot be less than '
                    f'last transaction date {last_transaction_date}.')
            
    
        
    @cached_property
    def asset_values(self) -> pd.Series:
        """Value of each asset."""
        return self._transactions.get_asset_values(self.date)
    
    
    @cached_property
    def current_stocks(self) -> pd.Series:
        """Current stocks and their asset values."""
        return self.asset_values[self.asset_values.values > 0]
    
    
    # @cached_property
    # def asset_net(self) -> float:
    #     """Net value of all assets."""
    #     funds = self.available_funds
    #     assets = self.asset_values.sum()
    #     return funds + assets
    
    
    @cached_property
    def asset_shares(self) -> pd.Series:
        """# of shares bought for each asset."""

        sdict = self._transactions._get_symbol_transactions_dict()
        date = self.date
        new = {}
        st : SymbolTransactions
        for symbol, st in sdict.items():
            date = np.datetime64(date)
            value = st.last_shares()
            new[symbol] = value
        return pd.Series(new)    

    
    @cached_property
    def return_ratios(self) -> pd.Series:
        """Ratio of return price since buy from zero shares."""
        return self._transactions.get_return_ratios(self.date)
    
    
    @cached_property
    def available_funds(self) -> float:
        return self._transactions.last_available_funds()
        

    @cached_property
    def equity(self) -> float:
        """Net value of all assets and funds."""
        funds = self.available_funds
        try:
            assets = self.asset_values.values.sum()
        except IndexError:
            assets = 0
        return funds + assets


class MarketState:
    def __init__(self, transactions: Transactions, date: np.datetime64):
        self._transactions = transactions
        self._stock_data = transactions.stock_data
        self.date = date
        
        
    @cached_property
    def stock_data(self) -> dict[pd.DataFrame]:
        """dict[DataFrame] : Dataframes for each symbol for the current increment."""
        return self._stock_data.filter_dates(end=self.date)  

    
    @cached_property
    def existing_symbols(self) -> list[str]:
        """Symbols which exist at the current date increment."""
        return self._stock_data.existing_symbols(self.date)
    
    
    @cached_property
    def unlisted_symbols(self) -> list[str]:
        """Symbols which do not exist at the current date increment."""
        return self._stock_data.unlisted_symbols(self.date)
    
    
    
    