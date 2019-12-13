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
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.models import Sequential, Model
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
import os.path

class DDPG:
    def __init__(self, input_dim, output_dim, **kwargs):
        assert isinstance(input_dim, int), "Input dim must be an integer"
        assert isinstance(output_dim, int), "Output dim must be an integer"

        ##### ARGS ######
        # Discount factor
        self.gamma = kwargs.get('gamma',0.90)
        # Learning rates for actor and critic
        self.actor_lr = kwargs.get('actor_lr',0.0001)
        self.critic_lr = kwargs.get('critic_lr',0.0001)
        # Smoothing factor for target networks (<<1)
        self.tau = kwargs.get('tau',0.01)
        # Memory size for experience replay
        self.memory_size = kwargs.get('memory_size',4000)
        # Size of minibatch (=how many samples to draw from experience per train)
        self.minibatch_size = kwargs.get('minibatch_size',256)
        # Probability of acting randomly (exploration)
        self.epsilon = kwargs.get('epsilon',0.9)
        # Decay of epsilon
        self.epsilon_decay = kwargs.get('epsilon_decay',0.99995)

        # Used for temporally correlated noise models
        self.last_action = None

        noise_models = {'ornstein_uhlenbeck' : self.__get_ornstein_uhlenbeck_action, 'gauss' : self.__get_gaussian_action}
        noise_model_name = kwargs.get('noise_model','gauss')

        self.noise_model = noise_models[noise_model_name]

        if not noise_model_name in noise_models:
            raise Exception("Unknown noise model {}. Choose from: {}".format(noise_model, noise_models))


        self.input_dim = input_dim
        self.output_dim = output_dim

        self.session = tf.Session()
        K.set_session(self.session)

        self.memory = deque()

        self.actor, self.actor_state_input = self.generate_actor()
        self.critic, self.critic_state_input, self.critic_action_input = self.generate_critic()

        # Store last action for noise-correlated 

        self.target_actor,_ = self.generate_actor()
        self.target_critic,_,_ = self.generate_critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        print("------------------- Actor model -------------------")
        print(self.actor.summary())
        print("------------------- Critic model ------------------")
        print(self.critic.summary())

        # Weights of actor model
        actor_weights = self.actor.trainable_weights

        # Variable that will hold the derivative dQ/da
        self.__critic_action_grad = tf.placeholder(tf.float32, [None, self.output_dim])

        # Gradient of loss function dJ/d(w_pi) = -dQ(s,a)/da * d(pi)/d(w_pi)
        self.__actor_gradient = tf.gradients(self.actor.output, actor_weights, -self.__critic_action_grad)
        grads = zip(self.__actor_gradient, actor_weights)

        self.__optimize_actor = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(grads)
        
        self.__critic_gradient = tf.gradients(self.critic.output, self.critic_action_input)

        # Initialize for later gradient calculations
        self.session.run(tf.initialize_all_variables())

    # Generate action. If in training, consider exploration
    def act(self, state, is_training=False):
        if is_training:
            action = self.actor.predict(np.array(state))
            if np.random.random() < self.epsilon:
                # Exploration
                if self.noise_model == self.__get_gaussian_action:
                    action += self.noise_model(action)
                elif self.noise_model == self.__get_ornstein_uhlenbeck_action and isinstance(self.last_action, np.ndarray):
                    action = self.noise_model(self.last_action)
            self.epsilon *= self.epsilon_decay
            self.last_action = action
        else:
            # Use target actor policy to act when not in training
            action = self.target_actor.predict(np.array(state))

        return action

    def __get_ornstein_uhlenbeck_action(self, action):
        action_t = action.reshape((self.output_dim,))

        tau = 1
        mu = 0.0
        sigma = 2.
        sigma_bis = sigma*np.sqrt(2./tau)

        new_action = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            new_action[i] = (action_t[i] + (mu - action_t[i])/tau + sigma_bis*np.random.randn())

        return new_action

    def __get_gaussian_action(self, action):
        new_action = action + np.random.randn(self.output_dim)/2.0
        return new_action

    def generate_actor(self):
        ###############################################################################
        ##  ACTOR: pi(s) --> [action1, action2, ...]
        ## Loss: J = -Q(s,pi(s|w_pi)) --> minimize loss == maximize expected reward
        ###############################################################################

        state_input = Input(shape=(self.input_dim,))
        state_h1 = Dense(500, activation='relu')(state_input)
        state_h2 = Dense(1000, activation='relu')(state_h1)
        state_h3 = Dense(500, activation='relu')(state_h2)

        actor_output = Dense(self.output_dim, activation='tanh')(state_h3)

        model = Model(input=state_input, output=actor_output)

        adam = Adam(learning_rate=self.actor_lr)

        model.compile(optimizer=adam,
        loss = 'mse',
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
        loss = 'mse',
        metrics=['mae']
        )

        return model, state_input, action_input

    def memorize(self, state, action, next_state, reward, done):
        self.memory.append([state, action, next_state, float(reward), float(done)])
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def sample_from_memory(self):
        return np.array(random.sample(self.memory, self.minibatch_size))

    def train(self, **kwargs):        
        if len(self.memory) < self.minibatch_size:
            return

        # Check args
        epochs = kwargs.get('epochs',1)
        
        samples = self.sample_from_memory()

        self.__train_critic(samples)
        self.__train_actor(samples)

        self.__update_target_networks()

    def __train_actor(self, samples):
        states = np.stack(samples[:,0]).reshape(samples.shape[0], -1)
        actions = np.stack(samples[:,1]).reshape(samples.shape[0], -1)
        next_states = np.stack(samples[:,2]).reshape(samples.shape[0], -1)
        rewards = np.stack(samples[:,3]).reshape(samples.shape[0], -1)

        predicted_actions = self.actor.predict(states)
        critic_gradient = self.session.run(self.__critic_gradient, feed_dict={self.critic_state_input: states, self.critic_action_input: predicted_actions})[0]

        # # DEBUG
        # state_t = states[0].reshape((-1, 3))
        # action_t = predicted_actions[0].reshape((-1, 1))
        # action_t2 = action_t + np.array([[0.001]])

        # Q = self.session.run(self.critic.output, feed_dict={self.critic_state_input: state_t, self.critic_action_input: action_t})
        # Q2 = self.session.run(self.critic.output, feed_dict={self.critic_state_input: state_t, self.critic_action_input: action_t2})


        self.session.run(self.__optimize_actor, feed_dict={self.actor_state_input: states, self.__critic_action_grad: critic_gradient})

    def __train_critic(self, samples):
        states = np.stack(samples[:,0]).reshape(samples.shape[0], -1)
        actions = np.stack(samples[:,1]).reshape(samples.shape[0], -1)
        next_states = np.stack(samples[:,2]).reshape(samples.shape[0], -1)
        rewards = np.stack(samples[:,3]).reshape(samples.shape[0], -1)
        dones = np.stack(samples[:,4]).reshape(samples.shape[0], -1) 

        predicted_actions = self.target_actor.predict(next_states)
        rewards += self.gamma*self.target_critic.predict([next_states, predicted_actions])*(1.0 - dones)
        self.critic.fit([states, actions], rewards, verbose=0)

    def __update_model_weights(self, model, target_model):
        weights = model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(weights)):
            target_weights[i] = self.tau*weights[i] + (1.0-self.tau)*target_weights[i]
        
        target_model.set_weights(target_weights)


    def __update_target_networks(self):
        self.__update_model_weights(self.actor, self.target_actor)
        self.__update_model_weights(self.critic, self.target_critic)

    def save(self, save_id):
        actor_save = save_id + "_actor_weight.h5"
        critic_save = save_id + "_critic_weight.h5"
        target_actor_save = save_id + "_target_actor_weight.h5"
        target_critic_save = save_id + "_target_critic_weight.h5"
        
        self.actor.save_weights(actor_save)
        self.target_actor.save_weights(target_actor_save)
        self.critic.save_weights(critic_save)
        self.target_critic.save_weights(target_critic_save)

    def load(self, save_id):
        actor_save = save_id + "_actor_weight.h5"
        critic_save = save_id + "_critic_weight.h5"
        target_actor_save = save_id + "_target_actor_weight.h5"
        target_critic_save = save_id + "_target_critic_weight.h5" 
        
        # Check if there's a saved model
        if self.save_exists(save_id):
            print("Loading model from {}, {}, {}, {}".format(actor_save, target_actor_save, critic_save, target_critic_save))
            self.actor.load_weights(actor_save)
            self.target_actor.load_weights(target_actor_save)
            self.critic.load_weights(critic_save)
            self.target_critic.load_weights(target_critic_save)
        else:
            raise Exception("Saved data with id {} not available.".format(save_id))
        
    def save_exists(self, save_id):
        actor_save = save_id + "_actor_weight.h5"
        critic_save = save_id + "_critic_weight.h5"
        target_actor_save = save_id + "_target_actor_weight.h5"
        target_critic_save = save_id + "_target_critic_weight.h5" 

        # Check if there's a saved model
        if os.path.isfile(actor_save) and os.path.isfile(critic_save) and os.path.isfile(target_actor_save) and os.path.isfile(target_critic_save):
            return True
        else:
            return False
        