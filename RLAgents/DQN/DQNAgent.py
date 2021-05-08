import numpy as np
from collections import deque
from .ReplayBuffer import ReplayBuffer
from .qnetworks import *
import sys
import tensorflow as tf
import gym
from ..networks import *


TRAIN_START = 500
BUFFER_LENGTH = 100_000
FINAL_EXPLORATION_FRAME = 200_000
MIN_EPS = 0.1
STEPS_PER_NETWORK_UPDATE = 4
DISCOUNT_FACTOR = 0.99
STEPS_PER_TARGET_UPDATE = 500
MINIBATCH_SIZE = 64
TRAINING_STEPS = 1_000_000
BETA_ANNEAL_STEPS = 2_000_000
LOG_STEPS = 500
LOG_FILE = "dqn_log.txt"


def log(text):
    with open(LOG_FILE, 'a') as f:
        f.write("{}\n".format(text))

# print = log


class DQNAgent:
    def __init__(self, env_fn, model_key, obs="Kinematics", resume=False,
                 use_double_dqn=False, use_dueling_dqn=False, use_priority=False,
                 normalize_reward_coeff=1):
        env = env_fn()
        self.env = env
        self.obs = obs

        if self.obs == "Kinematics":
            state_encoder = AttentionKinematicsEncoder
        elif self.obs == "Image":
            state_encoder = ImageEncoder

        self.total_steps_taken = 0
        self.steps_since_last_Q_update = 0
        self.steps_since_last_Target_update = 0
        self.steps_since_last_log = 0
        self.n_q_updates = 0
        self.n_target_updates = 0
        self.n_episodes = 0
        self.best_mean_reward = -np.inf
        self.use_double_dqn = use_double_dqn
        self.use_priority = use_priority
        if self.use_priority:
            self.beta0 = 0.4
            self.beta_anneal = (1 - self.beta0)/TRAINING_STEPS
            self.replay_buffer = ReplayBuffer(state_shape=env.observation_space.shape,
                                              action_shape=1,
                                              buffer_length=BUFFER_LENGTH,
                                              use_priority=self.use_priority,
                                              alpha=0.6,
                                              normalize_reward_coeff=normalize_reward_coeff,
                                              dtype=env.observation_space.dtype)
        else:
            self.replay_buffer = ReplayBuffer(state_shape=env.observation_space.shape,
                                              action_shape=1,
                                              buffer_length=BUFFER_LENGTH,
                                              use_priority=self.use_priority,
                                              normalize_reward_coeff=normalize_reward_coeff,
                                              dtype=env.observation_space.dtype)

        self.network_loss = deque(maxlen=1000)
        self.rewards = deque(maxlen=1000)
        self.q_network = FeedForwardPolicy(env.observation_space.shape,
                                           env.action_space.n,
                                           "dqn_models/q_network_{}/model.ckpt".format(model_key),
                                           dueling=use_dueling_dqn,
                                           scope="q_network",
                                           state_encoder=state_encoder)
        self.target_network = FeedForwardPolicy(env.observation_space.shape,
                                                env.action_space.n,
                                                "dqn_models/target_network_{}/model.ckpt".format(model_key),
                                                dueling=use_dueling_dqn,
                                                scope="target_network",
                                                state_encoder=state_encoder)
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q_network")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_network")
        self.update_target_ops = []
        for from_var, to_var in zip(q_vars, target_vars):
            self.update_target_ops.append(to_var.assign(from_var))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.q_network.set_session(self.sess)
        self.target_network.set_session(self.sess)
        self.update_target_network()
        if resume:
            self.q_network.load()
            self.update_target_network()

    def update_target_network(self):
        self.sess.run(self.update_target_ops)
        self.n_target_updates += 1

    def learn(self):
        while True:
            s = self.env.reset()
            episode_reward = 0

            while True:
                self.steps_since_last_log += 1

                if self.total_steps_taken > TRAIN_START:
                    eps = (1 - MIN_EPS) * (self.total_steps_taken/FINAL_EXPLORATION_FRAME)
                else:
                    eps = 1
                if np.random.random() > eps:
                    s_copy = np.copy(s)
                    a = np.argmax(self.q_network.predict(s_copy))
                else:
                    a = self.env.action_space.sample()
                s_, r, t, _ = self.env.step(a)
                episode_reward += r
                self.replay_buffer.append([s, a, r, s_, t])
                self.total_steps_taken += 1
                s = s_
                if self.total_steps_taken > TRAIN_START:
                    self.steps_since_last_Q_update += 1
                    if self.steps_since_last_Q_update >= STEPS_PER_NETWORK_UPDATE:
                        self.steps_since_last_Q_update = 0
                        self.train_q_network()
                        self.steps_since_last_Target_update += 1
                        if self.steps_since_last_Target_update >= STEPS_PER_TARGET_UPDATE:
                            self.update_target_network()
                            self.steps_since_last_Target_update = 0
                if t:
                    self.rewards.append(episode_reward)
                    self.n_episodes += 1
                    break
            if self.total_steps_taken > TRAINING_STEPS:
                self.update_target_network()
                self.write_log()
                break
            if self.steps_since_last_log >= LOG_STEPS:
                self.steps_since_last_log = 0
                self.write_log()

    def train_q_network(self):
        if self.use_priority:
            sample_idx, weights, states, actions, rewards, next_states, terminals = \
                self.replay_buffer.sample(MINIBATCH_SIZE, beta=self.beta0)
            self.beta0 += self.beta_anneal
            updated_priorities = []
        else:
            states, actions, rewards, next_states, terminals = \
                self.replay_buffer.sample(MINIBATCH_SIZE)
            weights = np.ones_like(terminals)
        outputs = []

        # for s, a, r, s_, t in zip(states, actions, rewards, next_states, terminals):
        for i in range(len(terminals)):
            s = states[i]
            r = rewards[i]
            a = actions[i]
            s_ = next_states[i]
            t = terminals[i]
            target = 0
            if t:
                target = r
            else:
                if self.use_double_dqn:
                    best_action = np.argmax(self.q_network.predict(s_))
                    target = r + DISCOUNT_FACTOR * self.target_network.predict(s_)[0][best_action]
                else:
                    target = r + DISCOUNT_FACTOR * np.max(self.target_network.predict(s_))
            outputs.append(target)
            if self.use_priority:
                updated_priorities.append(np.abs(target - self.q_network.predict(s)[0][a]))
        if self.use_priority:
            self.replay_buffer.update_priorities(sample_idx, updated_priorities)
        outputs = np.array(outputs)
        loss = self.q_network.train_step(states, actions, outputs, weights)
        self.network_loss.append(loss)
        self.n_q_updates += 1

    def write_log(self):
        mean_rewards = np.mean(self.rewards)
        eval_rewards = self.test_policy(games=10)
        mean_network_loss = np.mean(self.network_loss)
        text = "Num Steps: {}, Num Episodes: {}, Mean Rewards: {:.2f}, Evaluation Rewards: {:.2f}, Mean Network Loss: {:.2f}, Q Updates: {}, Target Updates: {}, Best Model: {:.2f}".format(
             self.total_steps_taken, self.n_episodes, mean_rewards, eval_rewards, mean_network_loss, self.n_q_updates, self.n_target_updates, self.best_mean_reward)
        print(text)
        if eval_rewards > self.best_mean_reward:
            self.best_mean_reward = eval_rewards
            self.q_network.save()
        self.target_network.save()

    def load_model(self, model_name):
        self.q_network.load(model_name)

    def test_policy(self, games=1, render=False, save=False):
        if save:
            from moviepy.editor import concatenate_videoclips, ImageClip
            import cv2
        images = []
        state_images = []
        rewards = []
        for _ in range(games):
            s = self.env.reset()
            episode_reward = 0
            while True:
                s_copy = np.copy(s)
                a = np.argmax(self.q_network.predict(s_copy))
                s_, r, t, _ = self.env.step(a)
                if render:
                    self.env.render()
                if save:
                    img = self.env.render('rgb_array')
                    images.append(img)
                    if self.obs == 'Image':
                        diff_images = s_[1:] - s_[:-1]
                        state_img = np.concatenate([st.T for st in s_], axis=1).astype('uint8')
                        diff_images = np.concatenate([d.T for d in diff_images], axis=1).astype('uint8')
                        state_img = np.concatenate([state_img, diff_images], axis=1)
                        # print(np.shape(state_img), np.shape(s_))
                        state_img = cv2.cvtColor(state_img, cv2.COLOR_GRAY2RGB)
                        # print(np.shape(state_img))
                        state_images.append(state_img)
                episode_reward += r
                s = s_
                if t:
                    break
            rewards.append(episode_reward)

            if save:
                clips = [ImageClip(img, duration=0.1) for img in images]
                final_clip = concatenate_videoclips(clips, method='compose')
                final_clip.write_videofile("video_dqn.mp4", fps=24)

                clips = [ImageClip(img, duration=0.1) for img in state_images]
                final_clip = concatenate_videoclips(clips, method='compose')
                final_clip.write_videofile("video_dqn_state_obs.mp4", fps=24)

        return np.mean(rewards)
