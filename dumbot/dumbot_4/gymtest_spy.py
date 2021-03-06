# -*- coding: utf-8 -*-
"""
Train a bot to trade on pure noise."""

import io

import numpy as np
import matplotlib.pyplot as plt

from globalcache import Cache
from dumbot.dqn.dqn import Settings, Trainer

from backtester import gym


env = gym.env_spy()
settings = Settings(
    epsilon_decay=.0007,
    epsilon_final=0,
    lr=.002
    )

cache = Cache(globals())

@cache.decorate
def train(numiters=2000, epsilon=1.0):
    trainer = Trainer(env, settings)
    df = trainer.train(numiters, epsilon)
    state_dict = trainer.target_net.state_dict()
    return state_dict, df


trainer = Trainer(env, settings)
state_dict, df0 = train(1000)
trainer.load_state_dict(state_dict)

scores, df = trainer.test(100)

plt.figure()
plt.subplot(2,1,1)
plt.plot(df0.index, df0['running_score'], label='running_score')
plt.legend()


plt.subplot(2,1,2)
plt.semilogy(df0.index, df0['loss'], label='loss')
plt.legend()

plt.figure()
df = df.sort_values(by='date')
plt.plot(df['date'], df['price'], '.--')

buy_locs = df['action'] == 1
date1 = df['date'][buy_locs]
price1 = df['price'][buy_locs]

plt.plot(date1, price1, 'o')