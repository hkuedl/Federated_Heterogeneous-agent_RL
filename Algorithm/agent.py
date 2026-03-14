import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam

from Algorithm.model import (Actor, Critic,Guard)
from Algorithm.memory import SequentialMemory
from Algorithm.random_process import OrnsteinUhlenbeckProcess
from Algorithm.util import *

criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()

class RL_agent(object):
    def __init__(self, nb_states, nb_actions, variant, device, seed):
        
        if seed > 0:
            self.seed(seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.index = variant['index']                  
        
        # Create Actor, Critic, and Guard Network
        net_cfg = {
            'hidden1':variant['hidden1'], 
            'hidden2':variant['hidden2'], 
            'init_w':variant['init_w']
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=variant['prate'])

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=variant['rate'])

        self.guard = Guard(self.nb_states, self.nb_actions, **net_cfg)
        self.guard_target = Guard(self.nb_states, self.nb_actions, **net_cfg)
        self.guard_optim  = Adam(self.guard.parameters(), lr=variant['rate'])

        hard_update(self.actor_target, self.actor)                        
        hard_update(self.critic_target, self.critic)
        hard_update(self.guard_target, self.guard)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=variant['rmsize'], window_length=variant['window_length'])
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=variant['ou_theta'], mu=variant['ou_mu'], sigma=variant['ou_sigma'])

        # Hyper-parameters
        self.batch_size = variant['bsize']
        self.tau = variant['tau']
        self.discount = variant['discount']
        self.depsilon = 1.0 / variant['epsilon']

        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        if USE_CUDA: 
            self.cuda(device)

    def update_policy(self):                    
        # Sample batch
        state_batch, action_batch, reward_batch, cost_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + self.discount*to_tensor(terminal_batch.astype(float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Prepare for the target g batch
        next_g_values = self.guard_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_g_values.volatile=False

        target_g_batch = to_tensor(cost_batch) + self.discount*to_tensor(terminal_batch.astype(float))*next_g_values
        
        # Guard update
        self.guard.zero_grad()

        g_batch = self.guard([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        gvalue_loss = criterion(g_batch, target_g_batch)
        gvalue_loss.backward()
        self.guard_optim.step() 

        # Actor update                                                
        self.actor.zero_grad()

        policy_loss = - (self.critic([to_tensor(state_batch),self.actor(to_tensor(state_batch))]) + 1*self.guard([to_tensor(state_batch),self.actor(to_tensor(state_batch))]))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.guard_target, self.guard, self.tau)

    def update_policy_KD(self, master, alpha):                  # policy update using knowledge distillation   
        # Sample batch
        state_batch, action_batch, reward_batch, cost_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + self.discount*to_tensor(terminal_batch.astype(float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        q_batch_master = master.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = alpha*criterion(q_batch, target_q_batch) + (1-alpha)*criterion(q_batch, q_batch_master)
        value_loss.backward()
        self.critic_optim.step()

        # Prepare for the target g batch
        next_g_values = self.guard_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_g_values.volatile=False

        target_g_batch = to_tensor(cost_batch) + self.discount*to_tensor(terminal_batch.astype(float))*next_g_values
        
        # Guard update
        self.guard.zero_grad()

        g_batch = self.guard([ to_tensor(state_batch), to_tensor(action_batch) ])
        g_batch_master = master.guard([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        gvalue_loss = alpha*criterion(g_batch, target_g_batch) + (1-alpha)*criterion(g_batch, g_batch_master)
        gvalue_loss.backward()
        self.guard_optim.step() 

        # Actor update                                                
        self.actor.zero_grad()

        policy_loss = - (self.critic([to_tensor(state_batch),self.actor(to_tensor(state_batch))]) + 1*self.guard([to_tensor(state_batch),self.actor(to_tensor(state_batch))]))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.guard_target, self.guard, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
        self.guard.eval()
        self.guard_target.eval()

    def cuda(self, device):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        self.guard.to(device)
        self.guard_target.to(device) 

    def observe(self, r_t, c_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, c_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)

        action_implementation = deepcopy(action)
        for i in self.index:
            action_implementation[i] = (action_implementation[i] + 1) / 2

        self.a_t = action
        return action_implementation

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        action_implementation = deepcopy(action)
        for i in self.index:
            action_implementation[i] = (action_implementation[i] + 1) / 2
        
        self.a_t = action
        return action_implementation

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

        self.guard.load_state_dict(
            torch.load('{}/guard.pkl'.format(output))
        )


    def save_model(self, output, key= None):
        if key is None:
            torch.save(
                self.actor.state_dict(),
                '{}/actor.pkl'.format(output)
            )
            torch.save(
                self.critic.state_dict(),
                '{}/critic.pkl'.format(output)
            )
            torch.save(
                self.guard.state_dict(),
                '{}/guard.pkl'.format(output)
            )
        else:
            torch.save(
                self.actor.state_dict(),
                '{}/actor{}.pkl'.format(output,key)
            )
            torch.save(
                self.critic.state_dict(),
                '{}/critic{}.pkl'.format(output,key)
            )
            torch.save(
                self.guard.state_dict(),
                '{}/guard{}.pkl'.format(output,key)
            )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)