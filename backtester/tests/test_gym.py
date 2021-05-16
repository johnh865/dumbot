# -*- coding: utf-8 -*-
import numpy as np
from backtester import gym


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
        print(env.backtest.strategy.asset_values)
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