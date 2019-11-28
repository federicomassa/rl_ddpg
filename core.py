######################################################################################################
# DDPG Algorithm --- Paper: Continuous control with Deep Reinforcement Learning - Lillicrap (2015)
#
# Create an actor pi(s) and a critic Q(s,a) 
# and a copy of these two networks that evolve slowly to improve stability (the 'target' networks)
# --> pi', Q'
######################################################################################################

import numpy as np
import tensorflow as tf
from collections import deque
import random

class MemorySample:
    def __init__(self, state, action, next_state, reward):
        assert state.shape == next_state.shape
        assert isinstance(reward, float)

        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

class DDPG:
    def __init__(self, input_dim, output_dim, **kwargs):
        assert isinstance(input_dim, int), "Input dim must be an integer"
        assert isinstance(output_dim, int), "Output dim must be an integer"

        # Check args
        self.gamma = kwargs.get('gamma',0.995)
        self.actor_lr = kwargs.get('actor_lr',0.0001)
        self.critic_lr = kwargs.get('critic_lr',0.0001)
        self.tau = kwargs.get('tau',0.01)
        self.memory_size = kwargs.get('memory_size',128)
        self.minibatch_size = kwargs.get('minibatch_size',32)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.session = tf.Session()

        self.memory = deque()

        self.actor, self.actor_state_input = self.generate_actor()
        self.critic, self.critic_state_input, self.critic_action_input = self.generate_critic()

        self.target_actor,_ = self.generate_actor()
        self.target_critic,_,_ = self.generate_critic()
        
        # Weights of actor model
        actor_weights = self.actor.trainable_weights

        # Variable that will hold the derivative dQ/da
        self.__critic_action_grad = tf.placeholder('float', [None, 1])

        # Gradient of loss function dJ/d(w_pi) = -dQ(s,a)/da * d(pi)/d(w_pi)
        self.__actor_gradient = tf.gradients(self.actor.output, actor_weights, -self.__critic_action_grad)
        
        self.__optimize_actor = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(self.__actor_gradient, actor_weights))

		self.__critic_gradient = tf.gradients(self.critic.output,
			self.critic_action_input)

    def generate_actor(self):
        ###############################################################################
        ##  ACTOR: pi(s) --> [action1, action2, ...]
        ## Loss: J = -Q(s,pi(s|w_pi)) --> minimize loss == maximize expected reward
        ###############################################################################

        state_input = Input(shape=self.input_dim)
        state_h1 = Dense(128, activation='relu')(state_input)
        model = Dense(self.output_dim, activation='linear')

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.actor_lr),
        loss = 'mean_squared_error',
        metrics=['mae']
        )

        return model, state_input

    def generate_critic(self):
        ###############################################################################
        ## CRITIC: Q(s,a) --> q-value
        ## Loss: J = (Q(s,a) - r_t - gamma*Q'(s_{t+1}, pi'(s_{t+1}))
        ###############################################################################

        state_input = Input(shape=self.input_dim)
		state_h1 = Dense(500, activation='relu')(state_input)
		state_h2 = Dense(1000)(state_h1)

		action_input = Input(shape=self.output_dim)
		action_h1    = Dense(500)(action_input)

		merged    = Concatenate()([state_h2, action_h1])
		merged_h1 = Dense(500, activation='relu')(merged)
		critic_output = Dense(1, activation='linear')(merged_h1)
		model  = Model(input=[state_input,action_input], output=critic_output)

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.critic_lr),
        loss = 'mean_squared_error',
        metrics=['mae']
        )

        return model, state_input, action_input

    def memorize(self, state, action, next_state, reward, done):
        memory.append(np.array([state, action, next_state, reward, done]))
        if len(memory) > self.memory_size:
            memory.popleft()

    def sample_from_memory(self, minibatch_size=self.minibatch_size):
        return np.array(random.sample(self.memory, minibatch_size))

    def train(self, **kwargs):        
        # Check args
        epochs = kwargs.get('epochs',50)
        
        samples = sample_from_memory()

        self.__train_actor(samples)
        self.__train_critic(samples)
        self.__update_target_networks()

    def __train_actor(self, samples):
        states, actions, next_states, rewards, _ = np.stack(samples[:,0]), 
            np.stack(samples[:,1]),
            np.stack(samples[:,2]),
            np.stack(samples[:,3])

        predicted_actions = self.actor.predict(states)
        critic_gradient = self.session.run(self.__critic_gradient, feed_dict={self.critic_state_input: current_states
        , self.critic_action_input: predicted_actions})

        self.session.run(self.__optimize_actor, feed_dict={self.actor_state_input: states, self.__critic_action_grad: critic_gradient})

    def __train_critic(self, samples):
        states, actions, next_states, rewards, dones = np.stack(samples[:,0]), 
            np.stack(samples[:,1]),
            np.stack(samples[:,2]),
            np.stack(samples[:,3]),
            np.stack(samples[:,4])

        predicted_actions = self.target_actor.predict(next_states)
        rewards += self.gamma*self.target_critic.predict([next_states, predicted_actions])*(1 - dones)
        self.critic.fit([states, actions], rewards)


    def __update_target_networks(self):
        target_actor_weights = self.tau*self.actor.get_weights() + (1.0-self.tau)*self.target_actor.get_weights()
        self.target_actor.set_weights(target_actor_weights)

        target_critic_weights = self.tau*self.critic.get_weights() + (1.0-self.tau)*self.target_critic.get_weights()
        self.target_critic.set_weights(target_critic_weights)

