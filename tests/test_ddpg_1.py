import pytest
from ddpg import *
import gym
import numpy as np
    
episode_length = 200
env = gym.make('Pendulum-v0')
ddpg = DDPG(env.observation_space.shape[0], env.action_space.shape[0])

def run(is_training=False):
    state = env.reset().reshape((1, ddpg.input_dim))
    for j in range(episode_length):
            action = ddpg.act(state, is_training)
            next_state,reward,done,_ = env.step(action)  
            next_state = next_state.reshape((1, ddpg.input_dim))

            if is_training:
                ddpg.memorize(state, action, next_state, reward, done)

            if is_training and j % int(episode_length/2.0) == 0:
                ddpg.train()
            if not is_training:
                env.render()
                print(action)

            state = next_state

            if done:
                break

def test_pendulum():
    num_episodes = 1000

    for i in range(num_episodes):
        render = (i%5 == 0)
        if render:
            run(False)
        else:
            run(True)
            
        
