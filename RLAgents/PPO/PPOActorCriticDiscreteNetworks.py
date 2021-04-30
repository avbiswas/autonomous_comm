import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class ActorDiscrete:

    def __init__(self, sess, n_states, n_actions, lr, state_encoder):

        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions
        self.sess = sess
        self.eps = 0.2
        self.isTraining = tf.placeholder(tf.bool, shape=[])

        self.input = tf.placeholder(shape=[None, *n_states],
                                    dtype=tf.float32, name="actor_state_input")
        self.old_logprobs = tf.placeholder(tf.float32, shape=[None, ],
                                           name="old_logprobs")
        self.action_input = tf.placeholder(tf.float32, shape=[None, ], name="action")
        # self.action = tf.expand_dims(self.action_input, -1)
        # self.old_logprobs_input = tf.expand_dims(self.old_logprobs, -1)
        self.advantages = tf.placeholder(tf.float32, shape=[None, ], name='advantages')
        self.attention_scores, self.context_vector = state_encoder(self.input)
        self.dense1 = tf.layers.dense(self.context_vector, 128, activation=tf.nn.relu,
                                      kernel_initializer=tf.orthogonal_initializer(1.0),
                                      name='actor_dense1')

        self.mu = tf.layers.dense(self.dense1, self.n_actions,
                                  kernel_initializer=tf.orthogonal_initializer(1.0),
                                  name='mu')

        self.policy = tfp.distributions.Categorical(tf.nn.softmax(self.mu, axis=1),
                                                    name='policy')
        self.predicted_action = tf.squeeze(self.policy.sample(1), axis=0)
        print(self.predicted_action)

        self.mu_greedy = tf.argmax(self.mu, axis=1)

        self.action_logprobs = self.policy.log_prob(self.predicted_action)
        self.new_logprobs = self.policy.log_prob(self.action_input)
        print(self.predicted_action, self.action_logprobs, self.new_logprobs)
        # exit()
        self.probability_ratio = tf.exp(self.new_logprobs - self.old_logprobs)
        self.p1 = (self.probability_ratio) * self.advantages

        self.p2 = tf.clip_by_value(self.probability_ratio,
                                   1 - self.eps, 1 + self.eps) * self.advantages
        self.surrogate_loss = -(tf.math.minimum(self.p1, self.p2))
        print(self.p1)
        self.entropy_loss = -0.01 * self.policy.entropy()
        self._loss = tf.reduce_mean(self.surrogate_loss) + tf.reduce_mean(self.entropy_loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss)

    def predict(self, state, test=False):
        if np.array(state).ndim == len(self.n_states):
            state = [state]
        if test:
            action = self.sess.run(self.mu_greedy,
                                   {self.input: state, self.isTraining: False,
                                    })
            logprobs = [[1]]
        else:
            action, logprobs, sf = self.sess.run([self.predicted_action, self.action_logprobs, tf.nn.softmax(self.mu, axis=1)],
                                             {self.input: state, self.isTraining: False,
                                              })

        return action, logprobs

    def get_probs(self, state):
        if np.array(state).ndim == len(self.n_states):
            state = [state]
        action = self.sess.run(self.mu,
                               {self.input: state, self.isTraining: False,
                                })
        return action

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
        print(action)
        _, loss = self.sess.run([self.train_op, self._loss],
                                {self.input: state, self.action_input: action,
                                 self.advantages: adv, self.isTraining: True,
                                 self.old_logprobs: old_logprobs})
        return loss


class CriticDiscrete:

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
