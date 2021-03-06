import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Actor():

    def __init__(self, sess, n_states, n_actions, lr, action_space_low, action_space_high, state_encoder,
                 learn_std=False):

        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.sess = sess
        self.eps = 0.2
        self.isTraining = tf.placeholder(tf.bool, shape=[])

        self.input = tf.placeholder(shape=[None, *n_states],
                                    dtype=tf.float32, name="actor_state_input")
        self.old_logprobs = tf.placeholder(tf.float32, shape=[None, self.n_actions],
                                           name="old_logprobs")
        self.action_input = tf.placeholder(tf.float32, shape=[None, self.n_actions], name="action")
        self.action = 2 * (self.action_input - self.action_space_low)/(self.action_space_high - self.action_space_low) - 1
        self.advantages = tf.placeholder(tf.float32, shape=[None, ], name='advantages')
        self.attention_scores, self.context_vector = state_encoder(self.input)
        self.dense1 = tf.layers.dense(self.context_vector, 128, activation=tf.nn.relu,
                                      kernel_initializer=tf.orthogonal_initializer(1.0),
                                      name='actor_dense1')

        self.mu = tf.layers.dense(self.dense1, self.n_actions, activation=tf.nn.tanh,
                                  kernel_initializer=tf.orthogonal_initializer(1.0),
                                  name='mu')
        if learn_std:
            self.sigma = tf.layers.dense(self.dense1, self.n_actions, activation=tf.nn.sigmoid,
                                         kernel_initializer=tf.orthogonal_initializer(1.0),
                                         name='sigma') * 0.25
        else:
            self.sigma = tf.Variable(initial_value=0.1*np.ones(self.n_actions, dtype=np.float32))

        self.sigma = tf.clip_by_value(self.sigma, 0.0005, 0.1)
        self.policy = tfp.distributions.Normal(self.mu, self.sigma, name='policy')
        '''
        self.predicted_action = (tf.clip_by_value(self.policy.sample(1),
                                                  action_space_low, action_space_high,
                                                  name='action_probs'))[0]
        '''
        self.predicted_action = tf.squeeze(self.policy.sample(1), axis=0)

        self.mu_rescaled = (self.mu + 1)/2 * (self.action_space_high - self.action_space_low) + self.action_space_low
        self.action_rescaled = (self.predicted_action + 1)/2 * (self.action_space_high - self.action_space_low) + self.action_space_low

        self.action_logprobs = self.policy.log_prob(self.predicted_action)
        self.new_logprobs = self.policy.log_prob(self.action)
        print(self.predicted_action, self.action_logprobs, self.new_logprobs)

        self.probability_ratio = tf.exp(self.new_logprobs - self.old_logprobs)
        self.advantages2 = tf.expand_dims(self.advantages, 1)
        self.p1 = (self.probability_ratio) * self.advantages2
        # self.clipped_probs = tf.clip_by_value(self.probability_ratio,
        #                     1 - self.eps, 1 + self.eps)
        self.p2 = tf.clip_by_value(self.probability_ratio,
                                   1 - self.eps, 1 + self.eps) * self.advantages2
        self.surrogate_loss = -(tf.math.minimum(self.p1, self.p2))
        self.entropy_loss = -0.01 * self.policy.entropy()
        self._loss = tf.reduce_mean(self.surrogate_loss) + tf.reduce_mean(self.entropy_loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss)

    def predict(self, state, test=False):
        if np.array(state).ndim == len(self.n_states):
            state = [state]
        if test:
            action = self.sess.run(self.mu_rescaled,
                                   {self.input: state, self.isTraining: False,
                                    })
            logprobs = [[1]]
        else:
            action, sigma, logprobs = self.sess.run([self.action_rescaled, self.sigma, self.action_logprobs],
                                             {self.input: state, self.isTraining: False,
                                              })

        return action, logprobs

    def learn(self, state, action, adv, old_logprobs):
        '''
        _, loss, new_logprobs, old_logprobs, probability_ratio, \
            p1, p2, a = self.sess.run([self.train_op, self._loss,
                                                       self.new_logprobs,
                                   self.old_logprobs, self.probability_ratio,
                                   self.p1, self.p2, self.advantages],
                                   {self.input: state, self.action: action,
                                    self.advantages: adv, self.isTraining: True,
                                    self.old_logprobs: old_logprobs})
        print("******")
        print("new_action_probs", new_logprobs)
        print("old_action_probs", old_logprobs)
        print("prratio", probability_ratio)
        print("Advantages", a)
        print("p1", p1)
        print("p2", p2)
        print("loss", loss)
        print("******")
        '''
        _, loss = self.sess.run([self.train_op, self._loss],
                                {self.input: state, self.action_input: action,
                                 self.advantages: adv, self.isTraining: True,
                                 self.old_logprobs: old_logprobs})
        return loss


class Critic:

    def __init__(self, sess, input_shape, lr, state_encoder):
        self.sess = sess
        self.lr = lr
        self.n_states = input_shape
        with tf.variable_scope("critic"):
            self.input = tf.placeholder(shape=[None, *input_shape],
                                        dtype=tf.float32, name="critic_state_input")
            self.isTraining = tf.placeholder(tf.bool, shape=[])
            self.target = tf.placeholder(tf.float32, shape=[None, 1], name="target")

            self.attention_scores, self.context_vector = state_encoder(self.input)

            self.dense1 = tf.layers.dense(self.context_vector, 128, activation=tf.nn.relu,
                                          kernel_initializer=tf.orthogonal_initializer(1.0))

            self.values = tf.layers.dense(self.dense1, 1,
                                          kernel_initializer=tf.orthogonal_initializer(1.0))
            self.output = tf.squeeze(self.values)
            self._loss = tf.reduce_mean(tf.squared_difference(self.values, self.target))
            self.train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss)

    def predict(self, state):
        if np.array(state).ndim == len(self.n_states):
            state = [state]
        return self.sess.run(self.output, feed_dict={self.input: state,
                                                     self.isTraining: False})

    def learn(self, state, target):
        _, loss = self.sess.run([self.train_step, self._loss],
                                feed_dict={self.input: state, self.target: target,
                                           self.isTraining: False})
        return loss


if __name__ == "__main__":
    tf.reset_default_graph()
    sess = tf.Session()
    actor = Actor(sess, 2, 1, 0.0001, [-1], [1])
    sess.run(tf.global_variables_initializer())
    print(actor.predict([[0, 4]]))
    actor.learn([[0, 4], [1, 5]], [[0.5], [0.4]], [1, 3], [[0], [1]], [[0.5], [0.6]])
