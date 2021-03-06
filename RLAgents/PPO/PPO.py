from .PPOActorCriticNetworks import Actor, Critic
from .PPOActorCriticDiscreteNetworks import ActorDiscrete, CriticDiscrete
from .memory_GAE import VectorizedMemory
import numpy as np
import tensorflow as tf
import gym
import sys
from sklearn.preprocessing import StandardScaler
import os
import argparse
import cv2
from .VectorizeEnv import VectorizedEnvs
from ..networks import *


def log(text):
    with open("auto.txt", 'a') as f:
        f.write("{}\n".format(text))


# print = log


class PPOAgent():

    def __init__(self, env_fn, resume=False, doScale=False, obs='Kinematics', model_key="dir"):
        self.sess = tf.Session()

        # self.env = env
        self.num_envs = 24
        self.env_fn = env_fn
        self.model_key = model_key
        self.obs = obs
        print(self.env_fn)

        self.env = VectorizedEnvs(self.env_fn, self.num_envs)

        if self.obs == "Kinematics":
            state_encoder = AttentionKinematicsEncoder
        elif self.obs == "Image":
            state_encoder = ImageEncoder
        self.doScale = doScale

        # if self.doScale:
        #    self.init_scaler()
        input_shape = self.env.observation_space.shape

        if isinstance(self.env.action_space, gym.spaces.Box):
            output_shape = self.env.action_space.shape[0]
            self.action_low = self.env.action_space.low
            self.action_high = self.env.action_space.high

            self.actor = Actor(self.sess, input_shape, output_shape, 5e-4, self.action_low,
                               self.action_high, state_encoder=state_encoder, learn_std=True)
            self.critic = Critic(self.sess, input_shape, 1e-3,  state_encoder=state_encoder)
        else:
            output_shape = self.env.action_space.n

            self.actor = ActorDiscrete(self.sess, input_shape, output_shape, 5e-4, state_encoder=state_encoder)
            self.critic = CriticDiscrete(self.sess, input_shape, 5e-4,  state_encoder=state_encoder)

        self.sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))
        self.gamma = 0.99
        self.lambd = 0.95
        self.memory = VectorizedMemory(self.num_envs)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join('./ppo_models/network_{}'.format(self.model_key),
                                            'model.ckpt')
        self.checkpoint_file_temp = os.path.join('./ppo_models/bkp_{}'.format(self.model_key),
                                                 'model.ckpt')

        self.batch_size = 128
        self.memory_buffer_length = 1280
        self.epochs = 10
        if resume:
            self.load_checkpoint()
        print("Batch Size: {}\nMemory Length: {}\nEpochs: {}".format(self.batch_size,
              self.memory_buffer_length, self.epochs))

    def remember(self, state, action, reward, state_next, done, logprobs, tlr):
        self.memory.remember(state, action, reward, state_next, done, logprobs, tlr)

    def act(self, state_dict, test=False):
        state = np.array([v for k, v in state_dict.items()])
        predicted_action, logprobs = self.actor.predict(state, test)
        predicted_action = {i: a for i, a in zip(state_dict.keys(), predicted_action)}
        logprobs = {i: lp for i, lp in zip(state_dict.keys(), logprobs)}
        return predicted_action, logprobs

    def test_play(self, gui=False, games=3, log=False, max_iter=None, save=False):
        if save:
            from moviepy.editor import concatenate_videoclips, ImageClip

        val_scores = []
        logs_all_games = []
        env = VectorizedEnvs(self.env_fn, games, observation=self.obs)
        logs = []
        state = env.reset()
        score = {i: 0 for i in state.keys()}
        steps = 0
        images = []
        state_images = []
        while True:
            if gui:
                env.render()

            action, _ = self.act(state, test=True)
            state_, reward, done, info = env.step(action)
            if save:
                img = env.render('rgb_array')
                images.append(img)
                # state_img = state_[0][-1].T.astype('uint8')
                # state_img = cv2.cvtColor(state_img, cv2.COLOR_GRAY2RGB)
                # state_images.append(state_img)
            for k, v in reward.items():
                score[k] += v

            state = state_
            logs.append(info)
            steps += 1
            if max_iter is not None:
                if steps > max_iter:
                    break
            if np.all([v for _, v in done.items()]):
                break
        if save:
            clips = [ImageClip(img, duration=0.1) for img in images]
            final_clip = concatenate_videoclips(clips, method='compose')
            final_clip.write_videofile("video.mp4", fps=24)

            # clips = [ImageClip(img, duration=0.25) for img in state_images]
            # final_clip = concatenate_videoclips(clips, method='compose')
            # final_clip.write_videofile("video_state_obs.mp4", fps=24)

        score_history = np.mean([s for _, s in score.items()])
        val_scores.append(score_history)
        logs_all_games.append(logs)

        return np.mean(val_scores), logs_all_games

    def learn(self):
        batch_state, batch_action, batch_reward, batch_state_next, batch_done, \
            batch_log_probs, batch_tlr = self.memory.getRecords()
        batch_state = np.array(batch_state)
        batch_action = np.array(batch_action)
        batch_log_probs = np.array(batch_log_probs)
        batch_size = len(batch_state)
        batch_advantage = np.array([0] * batch_size)
        batch_target = np.array([0] * batch_size)

        last_GAE = 0
        value_states = self.critic.predict(batch_state)
        value_state_last = self.critic.predict(batch_state_next[batch_size - 1])
        value_states = np.append(value_states, value_state_last)
        for i in reversed(range(batch_size)):
            value_state = value_states[i]
            value_state_next = value_states[i + 1]
            if batch_done[i]:
                if not batch_tlr[i]:
                    delta = batch_reward[i] - value_state
                else:
                    delta = batch_reward[i] + (self.gamma * value_state_next) - value_state
                last_GAE = delta
            else:
                delta = batch_reward[i] + (self.gamma * value_state_next) - value_state
                last_GAE = delta + self.lambd * self.gamma * last_GAE

            batch_advantage[i] = last_GAE
            batch_target[i] = delta + value_state

        batch_advantage = (batch_advantage - np.mean(batch_advantage))/(np.std(batch_advantage) + 1e-5)
        batch_target = np.expand_dims(batch_target, 1)
        length = len(batch_state)
        actor_loss = 0
        critic_loss = 0

        for _ in range(self.epochs):
            k = np.random.choice(length, self.batch_size, replace=False)
            actor_loss += self.actor.learn(batch_state[k], batch_action[k],
                                           batch_advantage[k], batch_log_probs[k])
            critic_loss += self.critic.learn(batch_state[k], batch_target[k])

        actor_loss /= self.epochs
        critic_loss /= self.epochs
        self.memory.clear()
        return actor_loss, critic_loss

    def load_checkpoint(self):
        print("...Loading checkpoint...")
        print(self.checkpoint_file)
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def init_scaler(self):
        self.scaler = StandardScaler()
        dataset = []
        for i in range(100000):
            dataset.append(self.env.observation_space.sample())
        self.scaler.fit(dataset)

    def scale(self, actual_state):
        return np.squeeze(self.scaler.transform([actual_state]))

    def play(self, test=False, save_model=False):

        if test:
            self.load_checkpoint()
            avg_score, _ = self.test_play(gui=True, games=10, log=False)
        else:
            n_games = 200000
            score_history = 0
            actor_losses = 0
            critic_losses = 0
            query_intervals = 1
            last_interval = 0
            max_score = -20000
            best_validation_score = -500
            for games in range(n_games):
                state = self.env.reset()
                score = {i: 0 for i in state.keys()}
                while True:
                    action, log_probs = self.act(state)

                    state_, reward, done, info = self.env.step(action)
                    time_limit_info = {k: v["time_limit_reached"] for k, v in info.items()}

                    self.remember(state, action, reward, state_, done, log_probs,
                                  time_limit_info)

                    state = state_
                    for k, v in reward.items():
                        score[k] += v
                    if np.all([v for _, v in done.items()]):
                        break

                score_history += np.mean([s for _, s in score.items()])
                max_score = max(score_history, max_score)

                last_interval += 1

                if self.memory.size() > self.memory_buffer_length:
                    actor_loss_, critic_loss_ = self.learn()
                    self.saver.save(self.sess, self.checkpoint_file_temp)
                    actor_losses += actor_loss_
                    critic_losses += critic_loss_
                    if last_interval >= query_intervals:
                        validation_score, _ = self.test_play(False, 25)
                        print("Games: {}, Score: {:.2f}, Max Score: {:.2f}, Validation: {:.2f},"
                              .format((games+1)*self.num_envs, score_history/last_interval,
                                      max_score, validation_score) +
                              " Actor Loss: {:.2f}, Critic Loss: {:.2f}".format
                              (actor_losses/last_interval, critic_losses/last_interval))
                        actor_losses = 0
                        critic_losses = 0
                        score_history = 0
                        max_score = -20000
                        last_interval = 0
                        if (validation_score > best_validation_score):
                            self.save_checkpoint()
                            best_validation_score = validation_score
                            print("Checkpoint Updated")
