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
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential, Model
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam

class DDPG:
    def __init__(self, input_dim, output_dim, **kwargs):
        assert isinstance(input_dim, int), "Input dim must be an integer"
        assert isinstance(output_dim, int), "Output dim must be an integer"

        ##### ARGS ######
        # Discount factor
        self.gamma = kwargs.get('gamma',0.995)
        # Learning rates for actor and critic
        self.actor_lr = kwargs.get('actor_lr',0.0001)
        self.critic_lr = kwargs.get('critic_lr',0.0001)
        # Smoothing factor for target networks (<<1)
        self.tau = kwargs.get('tau',0.01)
        # Memory size for experience replay
        self.memory_size = kwargs.get('memory_size',128)
        # Size of minibatch (=how many samples to draw from experience per train)
        self.minibatch_size = kwargs.get('minibatch_size',32)
        # Probability of acting randomly (exploration)
        self.epsilon = kwargs.get('epsilon',0.99)
        # Decay of epsilon
        self.epsilon_decay = kwargs.get('epsilon_decay',0.99)

        noise_models = {'ornstein_uhlenbeck' : self.__get_ornstein_uhlenbeck_action, 'gauss' : self.__get_gaussian_action}
        noise_model_name = kwargs.get('noise_model','ornstein_uhlenbeck')

        if not noise_model_name in noise_models:
            raise Exception("Unknown noise model {}. Choose from: {}".format(noise_model, noise_models))

        self.noise_model = noise_models[noise_model_name]

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.session = tf.Session()

        self.memory = deque()

        self.actor, self.actor_state_input = self.generate_actor()
        self.critic, self.critic_state_input, self.critic_action_input = self.generate_critic()

        # Store last action for noise-correlated 

        self.target_actor,_ = self.generate_actor()
        self.target_critic,_,_ = self.generate_critic()
        
        # Weights of actor model
        actor_weights = self.actor.trainable_weights

        # Variable that will hold the derivative dQ/da
        self.__critic_action_grad = tf.placeholder('float', [None, 1])

        # Gradient of loss function dJ/d(w_pi) = -dQ(s,a)/da * d(pi)/d(w_pi)
        self.__actor_gradient = tf.gradients(self.actor.output, actor_weights, -self.__critic_action_grad)
        
        self.__optimize_actor = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(self.__actor_gradient, actor_weights))

        self.__critic_gradient = tf.gradients(self.critic.output, self.critic_action_input)

    # Generate action. If in training, consider exploration
    def act(self, state, is_training=False):
        action = self.actor.predict(np.array(state))
        if is_training:
            if np.random.random() < self.epsilon:
                # Exploration
                action += self.noise_model(action)

        return action

    def __get_ornstein_uhlenbeck_action(self, action):
        tau = 0.05
        mu = 0.0
        sigma = 1.
        sigma_bis = sigma*np.sqrt(2./tau)

        new_action = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            new_action[i] = (action[i] + (mu - x[i])/tau + sigma_bis*np.random.randn()).clip(-1,1)

        return new_action

    def __get_gaussian_action(self, action):
        new_action = (action + np.random.randn(self.output_dim)).clip(-1,1)
        return new_action

    def generate_actor(self):
        ###############################################################################
        ##  ACTOR: pi(s) --> [action1, action2, ...]
        ## Loss: J = -Q(s,pi(s|w_pi)) --> minimize loss == maximize expected reward
        ###############################################################################

        state_input = Input(shape=(self.input_dim,))
        state_h1 = Dense(128, activation='relu')(state_input)
        actor_output = Dense(self.output_dim, activation='tanh')(state_h1)

        model = Model(input=state_input, output=actor_output)

        adam = Adam(learning_rate=self.actor_lr)

        model.compile(optimizer=adam,
        loss = 'mean_squared_error',
        metrics=['mae']
        )

        return model, state_input

    def generate_critic(self):
        ###############################################################################
        ## CRITIC: Q(s,a) --> q-value
        ## Loss: J = (Q(s,a) - r_t - gamma*Q'(s_{t+1}, pi'(s_{t+1}))
        ###############################################################################

        state_input = Input(shape=(self.input_dim,))
        state_h1 = Dense(500, activation='relu')(state_input)
        state_h2 = Dense(1000)(state_h1)

        action_input = Input(shape=(self.output_dim,))
        action_h1 = Dense(500)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(500, activation='relu')(merged)
        critic_output = Dense(1, activation='linear')(merged_h1)
        model = Model(input=[state_input,action_input], output=critic_output)

        adam = Adam(learning_rate=self.critic_lr)

        # Compile model
        model.compile(optimizer=adam,
        loss = 'mean_squared_error',
        metrics=['mae']
        )

        return model, state_input, action_input

    def memorize(self, state, action, next_state, reward, done):
        self.memory.append(np.array([state, action, next_state, reward, done]))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def sample_from_memory(self):
        return np.array(random.sample(self.memory, self.minibatch_size))

    def train(self, **kwargs):        
        if len(self.memory) < self.minibatch_size:
            return

        # Check args
        epochs = kwargs.get('epochs',50)
        
        samples = self.sample_from_memory()

        self.__train_actor(samples)
        self.__train_critic(samples)
        self.__update_target_networks()

    def __train_actor(self, samples):
        states = samples[:,0]
        actions = samples[:,1]
        next_states = samples[:,2]
        rewards = samples[:,3]

        print("BEGIN")
        print(states)
        print("END")
        print(states.shape)
        print(states[0].shape)
        predicted_actions = self.actor.predict(states)
        critic_gradient = self.session.run(self.__critic_gradient, feed_dict={self.critic_state_input: current_states
        , self.critic_action_input: predicted_actions})

        self.session.run(self.__optimize_actor, feed_dict={self.actor_state_input: states, self.__critic_action_grad: critic_gradient})

    def __train_critic(self, samples):
        states = samples[:,0]
        actions = samples[:,1]
        next_states = samples[:,2]
        rewards = samples[:,3]
        dones = samples[:,4] 

        predicted_actions = self.target_actor.predict(next_states)
        rewards += self.gamma*self.target_critic.predict([next_states, predicted_actions])*(1 - dones)
        self.critic.fit([states, actions], rewards)


    def __update_target_networks(self):
        target_actor_weights = self.tau*self.actor.get_weights() + (1.0-self.tau)*self.target_actor.get_weights()
        self.target_actor.set_weights(target_actor_weights)

        target_critic_weights = self.tau*self.critic.get_weights() + (1.0-self.tau)*self.target_critic.get_weights()
        self.target_critic.set_weights(target_critic_weights)

