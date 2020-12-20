# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:38:11 2020
UDACITY 
Deep Reinforcement Learning Nanodegree
Procject 1 - Navigation

@author: Ruben Kapp alias Token414
"""

from unityagents import UnityEnvironment
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dqn_agent import Agent
import torch

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))
print('\n')
# number of actions
this_action_size = brain.vector_action_space_size
print('Number of actions:', this_action_size)
print('\n')
# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
this_state_size = len(state)
print('\n')
print('States have length:', this_state_size)

''' INI the AGENT '''
agent = Agent(state_size = this_state_size, action_size=this_action_size, seed=0)
'''OR LOAD AGENT '''
#agent.qnetwork_local.load_state_dict(torch.load('checkpoint_2020-12-18_11-05-40.pth'))


# 1. Initialize the reply memory D with capacity N 
# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 64         # minibatch size
#    Done via initialisation fo the agent Replay memory D
#          agent.memory = ReplayBuffer(this_action_size, BUFFER_SIZE, BATCH_SIZE, seed)

# Initialize time step (for updating every UPDATE_EVERY steps)
agent.t_step = 0

# 2. initialize the action-value funtion with a random weight w (seed=0)
# 3. initialize the target action-value weight w'<-w
# Done via Deep Network in the model.py module within the agent class 
# by initialisation of the agent
#         # Q-Network
#           agent.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
#           agent.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)


# 4. for episode e <- 1 to M:
#      - initial input frame x1  
#      - preprare initial sate: S <-0([x1])
#      - for time step t <- 1 to T:
'''SAMPLE'''
#          - choose action A from state S using policy pi <- Epsilon-Greedy(^q(S,A,W))   
#          - take action A, observe reward R, and next input frame xt+1
#          - Prepare next state S' <- ([xt-2,xt-1, xt,xt+1])
#          - Store experience tuple (S,A,R,S') in reply memory D
#          - S <- S'
#
'''LEARN'''
#         - Obtain random minibatch of tuples (sj,aj,rj,sj+1) from D
#         -  Set target yj = rj +gamma *max(^q(sj+1,a,w'))
#         - Update: DELTA w = alpha(yj + ^q(sj,aj,w)) NABLA ^q(sj,aj,w)
#         - Every C Steps, reset: w' <- w
now = datetime.now()
d = now.strftime("%Y-%m-%d_%H-%M-%S")



M_EPISODES = 1000
#KEEP_Action_EVERY_N_STATES = 1
# init first action
#action = 1
#action_count = 1

# epsilon-greedy policy with decreasing rate of exploration
eps_start=1.0
eps_end=0.01
eps_decay=0.995

#t_step_all = 0                     # count all steps
scores = []                        # list containing scores from each episods
scores_window = deque(maxlen=100)  # last 100 scores
scores_mean = []                   # list that scores mean of last 100 episodes

eps = eps_start
for e in range(M_EPISODES):
    
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    
    while True:
        
        action = agent.act(state,eps = eps)         # select an action
        
        #update_actio = action_count % KEEP_Action_EVERY_N_STATES # keep action for n frames
        #if  update_actio==0:     
        #    action = next_action
        
        #action_count += 1
        env_info = env.step(int(action))[brain_name]        # send the action to the environment
        
        next_state = env_info.vector_observations[0]   # get the next state
        
        reward = env_info.rewards[0]                   # get the reward
        
        done = env_info.local_done[0]                  # see if episode has finished
        
        score += reward                                # update the score
        
        agent.step(state, action, reward, next_state, done) # do step 
                                                            # save experience in a reply buffer
                                                            # Learn every UPDATE_EVERY = 4       <- defined in dqn_agent.py 
        
        state = next_state                             # roll over the state to next time step
        
        #t_step_all +=1
    
        if done:                                       # exit loop if episode finished
            break
    
    # save most recent score
    scores_window.append(score)
    # save most recent score
    scores.append(score)   

    # decrease epsilon
    eps = max(eps_end, eps_decay*eps)               
    
    print('\rEPSILON {:.2f}'.format(eps), end="")
    print('\n')
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)), end="")
    print('\n')
    
    if e % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)))
        print('\n')
    if e >= 100:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)))
        scores_mean.append(np.mean(scores_window))
        print('\n')
    if np.mean(scores_window)>=13.00:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e-100, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_solved.pth')
        break




torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_'+d+'_End_Episode.pth')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# plotting the mean score calculated over the next 100 episodes for each current episode
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_mean)), scores_mean)
plt.title('DQN agent')
plt.ylabel('Score for the last 100 episodes')
plt.xlabel('Episode #')
plt.show()

env.close()