# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:34:33 2020

@author: Token (rk-innovation.com)
"""


from unityagents import UnityEnvironment
from dqn_agent import Agent
import torch

env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

this_action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
this_state_size = len(state)


''' INI the AGENT '''
agent = Agent(state_size = this_state_size, action_size = this_action_size, seed=0)
'''OR LOAD AGENT '''
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_solved.pth'))

M_EPISODES = 5
KEEP_Action_EVERY_N_STATES = 1

action_count = 0

for e in range(M_EPISODES):
    
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]             # get the current state
    score = 0                                           # initialize the score
    
    while True:
        
        next_action = agent.act(state,eps = 0)         #  select an action
        
        update_actio = action_count % KEEP_Action_EVERY_N_STATES # keep action for n frames
        if  update_actio==0:     
            action = next_action
        
        action_count += 1
        env_info = env.step(int(action))[brain_name]        # send the action to the environment
        
        next_state = env_info.vector_observations[0]   # get the next state
        done = env_info.local_done[0]                  # see if episode has finished
        
        state = next_state                             # roll over the state to next time step
        

    
        if done:                                       # exit loop if episode finished
            break
        
env.close()