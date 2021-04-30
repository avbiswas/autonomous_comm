import tensorflow as tf
import numpy as np


class FeedForwardPolicy:
    def __init__(self, input_shape, output_shape, model_path, scope, dueling=True,
                 state_encoder='DNN'):
        self.input_shape = input_shape
        self.output_dims = output_shape
        self.model_path = model_path
        self.scope = scope
        self.dueling = dueling
        self.state_encoder = state_encoder
        self.init_model()
        self.saver = tf.train.Saver()

    def init_model(self):
        with tf.variable_scope(self.scope):
            self.state_placeholder = tf.placeholder(tf.float32,
                                                    shape=[None, *self.input_shape])
            self.output_placeholder = tf.placeholder(tf.float32,
                                                     shape=[None])
            self.action_placeholder = tf.placeholder(tf.int32,
                                                     shape=[None])
            self.importance_weights = tf.placeholder(tf.float32,
                                                     shape=[None])

            if self.state_encoder == 'DNN':
                input = tf.layers.Dense(256, activation=tf.nn.relu)(self.state_placeholder)
                input = tf.layers.Dense(256, activation=tf.nn.relu)(input)
            else:
                _, input = self.state_encoder(self.state_placeholder)

            if not self.dueling:
                input = tf.layers.Dense(256, activation=tf.nn.relu)(input)
                self.output = tf.layers.Dense(self.output_dims)(input)
            else:
                value = tf.layers.Dense(256, activation=tf.nn.relu)(input)
                self.state_value = tf.layers.Dense(1)(value)
                advantage = tf.layers.Dense(256, activation=tf.nn.relu)(input)
                self.advantage = tf.layers.Dense(self.output_dims)(advantage)
                self.advantage_ = self.advantage - tf.reduce_mean(self.advantage, axis=-1,
                                                                  keepdims=True)
                self.output = self.state_value + self.advantage_

            self.action_one_hot = tf.one_hot(self.action_placeholder, self.output_dims)
            self.q_val_action = tf.reduce_sum(self.output * self.action_one_hot, 1)
            self.mse_loss = tf.reduce_mean(self.importance_weights *
                                           tf.square(self.output_placeholder - self.q_val_action))
            self._params = tf.trainable_variables(scope=self.scope)
            print("OUTPUT: ", self.output)
            print("ACTION OHE: ", self.action_one_hot)
            print("Q_VAL: ", self.q_val_action)
            print("MSE: ", self.mse_loss)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0005,
                                                       momentum=0.95,
                                                       epsilon=0.01)
            # gradients = self.optimizer.compute_gradients(self.mse_loss, self._params)
            # gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            gradients, variables = zip(*self.optimizer.compute_gradients(self.mse_loss,
                                                                         self._params))

            gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5.0)
                         for gradient in gradients]
            self.opt = self.optimizer.apply_gradients(zip(gradients, variables))

    def set_session(self, sess):
        self.sess = sess

    def predict(self, state):
        if np.array(state).ndim == len(self.input_shape):
            state = [state]
        with tf.variable_scope(self.scope):
            output = self.sess.run(self.output,
                                   {self.state_placeholder: state})

        return output

    def train_step(self, states, actions, outputs, weights):
        if np.array(states).ndim == len(self.input_shape):
            states = [states]

        actions = np.array(actions).astype('int32')
        # print(weights)
        with tf.variable_scope(self.scope):
            _, loss = self.sess.run([self.opt, self.mse_loss],
                                    {self.state_placeholder: states,
                                     self.action_placeholder: actions,
                                     self.output_placeholder: outputs,
                                     self.importance_weights: weights})
        return loss

    def save(self):
        self.saver.save(self.sess, self.model_path)

    def load(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.saver.restore(self.sess, self.model_path)
