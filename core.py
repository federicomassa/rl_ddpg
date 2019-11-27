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

class DDPG:
    def __init__(self, input_dim, output_dim, **kwargs):
        assert isinstance(input_dim, int), "Input dim must be an integer"
        assert isinstance(output_dim, int), "Output dim must be an integer"

        # Check args
        self.actor_lr = kwargs.get('actor_lr',0.0001)
        self.critic_lr = kwargs.get('critic_lr',0.0001)
        self.tau = kwargs.get('tau',0.01)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.session = tf.Session()

        self.memory = deque()

        ###############################################################################
        ##  ACTOR: pi(s) --> [action1, action2, ...]
        ## Loss: J = -Q(s,pi(s|w_pi)) --> minimize loss == maximize expected reward
        ###############################################################################

        self.actor = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='linear')
        ])

        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.actor_lr),
        loss = 'mean_squared_error',
        metrics=['mae']
        )
        
        # Weights of actor model
        actor_weights = self.actor.trainable_weights

        # Variable that will hold the derivative dQ/da
        self.__critic_action_grad = tf.placeholder('float', [None, 1])

        # Gradient of loss function dJ/d(w_pi) = -dQ(s,a)/da * d(pi)/d(w_pi)
        self.__actor_gradient = tf.gradients(self.actor.output, self.__actor_weights, -self.__critic_action_grad)
        
        self.__optimize_actor = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(self.__actor_gradient, actor_weights))

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
		self.critic  = Model(input=[state_input,action_input], output=critic_output)

        # Compile model
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.critic_lr),
        loss = 'mean_squared_error',
        metrics=['mae']
        )

		self.__critic_gradient = tf.gradients(self.critic.output,
			action_input) # where we calcaulte de/dC for feeding above

    def train(self, **kwargs):        
        # Check args
        epochs = kwargs.get('epochs',50)
        
        self.__train_actor()
        self.__train_critic()

    def __train_actor(self):
        
