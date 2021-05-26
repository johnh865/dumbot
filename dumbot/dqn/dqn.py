# -*- coding: utf-8 -*-
import pdb 
from typing import NamedTuple
from collections import namedtuple, deque
import random

import torch
from torch import nn
import torch.optim as optim

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from dumbot.dqn.model import QNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
class Settings(NamedTuple):
    
    gamma: float = 0.99
    batch_size: int = 32
    lr: float = .005
    initial_exploration: int = 1000
    goal_score: float = None
    log_interval: int = 10
    update_target: int = 10
    replay_memory_capacity: int = 1000
    epsilon_decay: float = .002
    epsilon_final: float = 0.10
    
    
    
    
class Transition(NamedTuple):
    state : torch.Tensor
    next_state : torch.Tensor
    action : torch.Tensor
    reward : torch.Tensor
    mask : torch.Tensor
    
    
class Memory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
    
    
    def push(self, state, next_state, action, reward, mask: int):
        """Append to memory"""
        self.memory.append(Transition(state, next_state, action, reward, mask))
        
        
    def sample(self, batch_size: int):
        """Retrieve a random sample of the memory"""
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    
    def __len__(self):
        return len(self.memory)
    
    
    
def get_action(state, target_net: nn.Module, epsilon: float, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)
    
    
def update_target_model(online_net: nn.Module, target_net: nn.Module):
    target_net.load_state_dict(online_net.state_dict())
    
    

class Trainer:
    def __init__(self, env, settings: Settings, max_score=None, ):
        self.settings = settings
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        print('state size:', num_inputs)
        print('action size:', num_actions)
    
        online_net = QNet(num_inputs, num_actions)    
        target_net = QNet(num_inputs, num_actions)    
        update_target_model(online_net, target_net)
        
        optimizer = optim.Adam(online_net.parameters(), lr=self.settings.lr)
        writer = SummaryWriter('logs')
        online_net.to(device)
        target_net.to(device)

        memory = Memory(self.settings.replay_memory_capacity)
        
        self.online_net = online_net
        self.target_net = target_net
        self.optimizer = optimizer 
        self.memory = memory
        self.env = env
        self.writer = writer
        self.num_actions = num_actions
        self.max_score = max_score
        
        
    def train(self, numiter: int, epsilon=1.0):
        online_net = self.online_net
        target_net = self.target_net
        online_net.train()
        target_net.train()        
        
        optimizer = self.optimizer
        memory = self.memory
        writer = self.writer
        env = self.env
        num_actions = self.num_actions
        max_score = self.max_score
        
        # Retrieve settings
        initial_exploration = self.settings.initial_exploration
        batch_size = self.settings.batch_size
        update_target = self.settings.update_target
        epsilon_decay = self.settings.epsilon_decay
        epsilon_final = self.settings.epsilon_final
        log_interval = self.settings.log_interval
        goal_score = self.settings.goal_score
        
        running_score = 0
        # epsilon = 1.0
        steps = 0
        loss = 0        
        
        running_score_list = []
        loss_list = []
        for e in range(numiter):
            done = False
            score = 0
            state = env.reset()
            state = torch.Tensor(state).to(device)
            state = state.unsqueeze(0)
            
            while not done:
                steps += 1
                action = get_action(state, target_net, epsilon, env)
                next_state, reward, done, _ = env.step(action)
                next_state = torch.Tensor(next_state).unsqueeze(0)

                mask = 0 if done else 1
                score += reward
                action_one_hot = np.zeros(num_actions)
                action_one_hot[action] = 1
                
                memory.push(state, next_state, action_one_hot, reward, mask)
                state = next_state
                
                if steps > initial_exploration:

                    batch = memory.sample(batch_size)
                    loss = QNet.train_model(online_net,
                                            target_net,
                                            optimizer,
                                            batch)                
                    if steps % update_target == 0:
                        update_target_model(online_net, target_net)    
                
                if max_score:
                    if score >= max_score:
                        done = True
                        
            epsilon -= epsilon_decay
            epsilon = max(epsilon, epsilon_final)
                        
            running_score = 0.95 * running_score + .05 * score
            if e % log_interval == 0:
                s = (
                    f'{e} episode | score: {running_score:.2f}'
                    f' | epsilon: {epsilon:.2f}'
                    )
                print(s)
                running_score_list.append(running_score)
                loss_list.append(float(loss))
                writer.add_scalar('log/score', float(running_score), e)
                writer.add_scalar('log/loss', float(loss), e)            
            if goal_score is not None:
                if running_score > goal_score:
                    print('goal reached.')
                    break
        
        d = {}
        d['running_score'] = np.array(running_score_list)
        d['loss'] = np.array(loss_list)
        return pd.DataFrame(d)
                
            
    def test(self, numiter: int, epsilon: float=0):
        """Test a trained model."""
        target_net = self.target_net
        env = self.env
        max_score = self.max_score
                
        info_list = []
        
        actions = []
        rewards = []
        scores = []
        for e in range(numiter):
            done = False
            score = 0
            state = env.reset()
            state = torch.Tensor(state).to(device)
            state = state.unsqueeze(0)
            while not done:
                action = get_action(state, target_net, epsilon, env)
                next_state, reward, done, info = env.step(action)

                actions.append(action)                
                rewards.append(reward)
                info_list.append(info)
                
                state = torch.Tensor(next_state).unsqueeze(0)          
                score += reward
                if max_score:
                    if score >= max_score:
                        done = True
            scores.append(score)
            
        df = pd.DataFrame.from_records(data=info_list)
        df['action'] = actions
        df['reward'] = rewards
        return scores, df
    
    
    def save(self, path: str):
        model = self.target_net
        torch.save(model.state_dict(), path)
        
    
    def load_state_dict(self, d):
        model = self.target_net
        model.load_state_dict(d)
        model = self.online_net
        model.load_state_dict(d)        
        

    def load(self, path: str):
        model = self.target_net
        model.load_state_dict(torch.load(path))
        model = self.online_net
        model.load_state_dict(torch.load(path))        


if __name__ == '__main__':
    from backtester import gym
    env = gym.env_sin20_growth()
    settings = Settings(
        epsilon_decay=.0007,
        lr=.002
        )
    trainer = Trainer(env, settings)
    trainer.train(1500)
    
    