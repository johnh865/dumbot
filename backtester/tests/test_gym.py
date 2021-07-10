# -*- coding: utf-8 -*-
import pdb 
import numpy as np
from backtester import gym
from backtester.utils import floor_to_date

def test1():
    """Test to make sure asset calculations are correct."""
    env = gym.env_sin20_growth()
    env.reset()
    symbol = env.symbol
    
    transactions = env.backtest.transactions
    
    out = []
    rewards = []
    actions = [0, 1, 0, 1, 1, 0, 1]
    for action in actions:
        output = env.step(action)
        print(env.backtest.strategy.state.equity)
        out.append(output)
        reward = output[1]
        rewards.append(reward)
    
    
    df = transactions.dataframe
    equity = df['equity'].values
    rewards2 = equity[1:] - equity[:-1]
    
    asset_history = transactions.asset_history
    
    st = transactions.get_symbol_transactions(symbol)
    
    ah = asset_history
    assert np.all(ah[symbol] + ah['available_funds'] == ah['equity'])
    
    

def test2():
    """Test to make sure dates align for step and reset"""
    env = gym.env_noise(0)
    symbol = env.symbol
    
    
    indicator = env.indicators.dataframes[symbol]
    df_stock = env.backtest.stock_data.dataframes[symbol]
    
    
    # Start at index=5000, 2009-10-30
    obs = env.reset(time_index=5000)
    ind = obs[0:-1]
    out = indicator.iloc[env.time_index]
    date1 = df_stock.index[env.time_index]
    date_env = indicator.index[env.time_index]
    close = df_stock.iloc[env.time_index]
    
    
    assert np.all(out == ind)
    assert date1 == date_env
    
    # Increment, perform action on 2009-10-30 ... NO BUY
    # Retrieve observation for next date 2009-11-02.
    date = np.datetime64('2009-10-30')
    date_next = np.datetime64('2009-11-02')
    obs1, reward1, done1, info1 = env.step(1)
    ind1 = obs1[0 : -1]
    close1 = df_stock.loc[date_next]
    
    assert np.isclose(reward1, -.2)
    assert np.all(indicator.loc[date_next] == ind1)
    assert floor_to_date(env.backtest.strategy.date) == date
    
    
    # # Buy on 2009-11-02
    # Retrieve observation for next date 2009-11-03
    date = np.datetime64('2009-11-02')
    date_next = np.datetime64('2009-11-03')
    obs2, reward2, done2, info2 = env.step(1)
    ind2 = obs2[0 : -1]
    
    assert np.all(indicator.loc[date_next] == ind2)
    assert floor_to_date(env.backtest.strategy.date) == date
    
    obs3, reward3, done3, info3 = env.step(1)
    
    
    obs4, reward4, done4, info4 = env.step(0)

    df = env.backtest.transactions.dataframe
    assert 100 + reward1 + reward2 + reward3 + reward4 == df['equity'].iloc[-1]

    
    return



# def test_return_ratio():
#     env = gem.env_noise(0)
    
    


if __name__ == '__main__':
    test1()
    test2()