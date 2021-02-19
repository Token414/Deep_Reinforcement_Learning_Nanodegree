# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:34:33 2020

@author: Token (rk-innovation.com)
"""

import numpy as np
from unityagents import UnityEnvironment
from ddpg_agent import Agent
import torch


env = UnityEnvironment(file_name='Tennis_Windows_x86_64\Tennis.exe')


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

this_action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
this_state_size = len(state)


''' INI the AGENT '''

agent = Agent(state_size = this_state_size, action_size = this_action_size,random_seed=0)
'''OR LOAD AGENT '''


agent.actor_local.load_state_dict(torch.load('trained_agent/actor_weights.pth'))
agent.critic_local.load_state_dict(torch.load('trained_agent/critic_weights.pth'))

                
M_EPISODES = 10
max_t = 1000

num_agents = len(env_info.agents)

for i_episode in range(1, M_EPISODES+1):
        env_info = env.reset(train_mode=False)[brain_name]      # reset environment
        states = env_info.vector_observations                   # get current state for each agent      
        scores = np.zeros(num_agents)                           # initialize score for each agent
        agent.reset()

        for t in range(max_t):
            actions = agent.act(states, add_noise=True)         # select an action
            env_info = env.step(actions)[brain_name]            # send actions to environment
            next_states = env_info.vector_observations          # get next state
            rewards = env_info.rewards                          # get reward
            dones = env_info.local_done                         # see if episode has finished
            # save experience to replay buffer, perform learning step at defined interval
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)
            states = next_states
            scores += rewards        
            if np.any(dones):                                   # exit loop when episode ends
                break

    
        if done:                                       # exit loop if episode finished
            #break
            pass
        
env.close()