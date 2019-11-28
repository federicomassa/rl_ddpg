import pytest
from ddpg import *
    
def test_pendulum():
    import gym
    import numpy as np

    env = gym.make('Pendulum-v0')
    ddpg = DDPG(env.observation_space.shape[0], env.action_space.shape[0])

    episode_length = 200
    num_episodes = 1000

    for i in range(num_episodes):
        state = env.reset()

        render = (i%5 == 0)
        for j in range(episode_length):
            print(np.array(state).reshape((1,ddpg.input_dim)))
            print(state[0])
            action = ddpg.act(np.array(state).reshape((1, env.observation_space.shape[0])))
            next_state,reward,done,_ = env.step(action)

            ddpg.memorize(np.array(state).reshape((1, ddpg.input_dim)), np.array(action).reshape((1, ddpg.output_dim)), np.array(next_state).reshape((1, ddpg.input_dim)), reward, done)
            state = next_state

            if j % 5 == 0:
                ddpg.train()

            if render:
                env.render()

            if done:
                break

            
        
