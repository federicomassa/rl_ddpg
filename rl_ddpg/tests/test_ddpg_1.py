import pytest
from ddpg import *
import gym
import numpy as np
import os.path
import pdb
pdb.set_trace()

episode_length = 200
env = gym.make('Pendulum-v0')
ddpg = DDPG(env.observation_space.shape[0], env.action_space.shape[0], noise_model='ornstein_uhlenbeck')

# Check if there's a saved model
if os.path.isfile("actor_weights.h5") and os.path.isfile("critic_weights.h5"):
    print("Loading model from {}, {}".format("actor_weights.h5", "critic_weights.h5"))
    ddpg.actor.load_weights("actor_weights.h5")
    ddpg.target_actor.load_weights("actor_weights.h5")
    ddpg.critic.load_weights("critic_weights.h5")
    ddpg.target_critic.load_weights("critic_weights.h5")

def run(is_training=False):
    state = env.reset().reshape((1, ddpg.input_dim))
    tot_reward = 0
    for j in range(episode_length):
            action = ddpg.act(state, is_training)
            next_state,reward,done,_ = env.step(action*2)  
            next_state = next_state.reshape((1, ddpg.input_dim))
            tot_reward += reward

            if j == episode_length - 1:
                print("Final reward: {}".format(tot_reward))

            if is_training:
                ddpg.memorize(state, action, next_state, reward, done)

            if is_training and j % 5 == 0:
                ddpg.train()
            if not is_training:
                env.render()

            state = next_state

            if done:
                break

def test_pendulum():
    num_episodes = 1000
    for i in range(num_episodes):
        print("Trial: ", i)
        render = (i%10 == 0)
        if render:
            run(False)
        else:
            run(True)

        if i%10 == 0:
            ddpg.actor.save_weights('actor_weights.h5')
            ddpg.critic.save_weights('critic_weights.h5')
            
        
