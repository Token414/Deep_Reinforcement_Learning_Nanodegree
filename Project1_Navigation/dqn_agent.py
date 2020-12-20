# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:49:54 2020
UDACITY COURSE Deep Reinforcement Learning
DQN Agent
@author: Token
"""

import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
UPDATE_TARGET_EVERY = 110



# Check Cuda versions and display GPU
from torch import cuda
print(cuda.is_available())
print(cuda.device_count())
#print(cuda.get_device_name(cuda.current_device()))
print()


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # use cpu


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        
        self.loss = []
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_step_target = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

               
    def learn(self, experiences, gamma):
        """
        In order to calculate the loss (TD-error), we need to calculate 2 values:           
        ===========
        1. Target values 
        2. current Q-values for each state.
        
        Now, in order to break the correlation between current Q-values and the
        target Q-values, we require two set of weights. One is called local or
        online weights that are updated at every UPDATE_ONLINE_EVERY steps while
        the target weights are updated with the current online weights at
        UPDATE_TARGET_EVERY steps.
        
        """
        states, actions, rewards, next_states, dones = experiences
        
        
        _, greedy_actions = torch.max(self.qnetwork_local(next_states).detach(), dim=1)
        greedy_actions = torch.unsqueeze(greedy_actions, 1)
        # we select the action values corresponding to the greedy_actions we selected from
        # the local/online network
        
        q_greedy_targets = self.qnetwork_target(next_states).gather(1, greedy_actions)
        
        q_greedy_targets = (1 - dones) * q_greedy_targets
        q_targets = rewards + gamma * (q_greedy_targets)
        
        q_current_est = self.qnetwork_local(states).gather(1, actions)
        
        
        # backpropagation step
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_targets, q_current_est)
        
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss)
        #print('Loss: ' + str(loss))
        # -----------------update the target network----------------- #
        self.t_step_target = (self.t_step_target + 1) % UPDATE_TARGET_EVERY
        
        if self.t_step_target == 0:
#             self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
            self.update_target(self.qnetwork_local, self.qnetwork_target)
            

    def update_target(self, online_model, target_model):
        #print('Update Target!')
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(online_param.data)
            

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self,  buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        #self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        """
        The following code decouples all components (s, a, r, s', d) from the
        experiences list and joins all components of the same type together
        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)