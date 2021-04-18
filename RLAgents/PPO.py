from .PPOActorCriticNetworks import Actor, Critic
from .memory_GAE import Memory
import numpy as np
import tensorflow as tf
import gym
import sys
from sklearn.preprocessing import StandardScaler
import os
import argparse
import cv2


def log(text):
    with open("auto.txt", 'a') as f:
        f.write("{}\n".format(text))


print = log


class PPOAgent():

    def __init__(self, env, resume=False, doScale=False, dir='chkpt'):
        self.sess = tf.Session()
        self.env = env
        self.doScale = doScale
        if self.doScale:
            self.init_scaler()
        print("Observation Space Shape: ", env.observation_space)
        print("Action Space Shape, Low, High: ", env.action_space,
                                                 env.action_space.low,
                                                 env.action_space.high)
        input_shape = env.observation_space.shape
        output_shape = env.action_space.shape[0]
        print(input_shape, input_shape[0], output_shape)
        self.action_low = env.action_space.low[0]
        self.action_high = env.action_space.high[0]
        # tf.reset_default_graph()
        self.actor = Actor(self.sess, input_shape, output_shape, 5e-4, self.action_low,
                           self.action_high, learn_std=True)
        self.critic = Critic(self.sess, input_shape, 1e-3)
        self.sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))
        self.gamma = 0.99
        self.lambd = 0.95
        self.memory = Memory()
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join('./{}'.format(dir),
                                            '{}_network.ckpt'.format("car"))
        self.batch_size = 64
        self.memory_buffer_length = 640
        self.epochs = 20
        if resume:
            self.load_checkpoint()
        print("Batch Size: {}\nMemory Length: {}\nEpochs: {}".format(self.batch_size,
              self.memory_buffer_length, self.epochs))

    def remember(self, state, action, reward, state_next, done, logprobs):
        self.memory.remember(state, action, reward, state_next, done, logprobs)

    def act(self, state, test=False):
        state = np.array(state)
        predicted_action, logprobs = self.actor.predict(state[np.newaxis, :], test)
        if self.action_low > 1:
            return predicted_action * self.action_low, logprobs
        else:
            return predicted_action, logprobs

    def test_play(self, gui=False, games=3, log=False, max_iter=None, save=False):
        val_scores = []
        logs_all_games = []
        for game in range(games):
            logs = []
            score = 0
            state = self.env.reset()
            steps = 0
            images = []
            while True:
                if gui:
                    import time
                    self.env.render()
                    # time.sleep(1/240)
                if self.doScale:
                    state = self.scale(state)
                action, _ = self.act(state, test=True)
                state_, reward, done, info = self.env.step(action)
                if save:
                    img = self.env.render('rgb_array')
                    cv2.imshow("", img)
                    cv2.waitKey(1)
                    images.append(img)
                score += reward
                state = state_
                logs.append(info)
                steps += 1
                if max_iter is not None:
                    if steps > max_iter:
                        break
                if done:
                    break
            if save:
                from numpngw import write_apng
                write_apng('anim_{}.png'.format(game), images, delay=20)

            val_scores.append(score)
            logs_all_games.append(logs)

        return np.mean(val_scores), logs_all_games

    def learn(self):
        batch_state, batch_action, batch_reward, batch_state_next, batch_done, batch_log_probs = \
            self.memory.getRecords()
        batch_size = len(batch_state)
        batch_advantage = [0] * batch_size
        batch_target = [0] * batch_size

        last_GAE = 0
        value_states = self.critic.predict(batch_state)
        value_state_last = self.critic.predict(batch_state_next[batch_size - 1])
        value_states = np.append(value_states, value_state_last)
        for i in reversed(range(batch_size)):
            value_state = value_states[i]
            value_state_next = value_states[i + 1]
            if batch_done[i]:
                delta = batch_reward[i] - value_state
                last_GAE = delta
            else:
                delta = batch_reward[i] + (self.gamma * value_state_next) - value_state
                last_GAE = delta + self.lambd * self.gamma * last_GAE

            batch_advantage[i] = last_GAE
            batch_target[i] = delta + value_state

        batch_advantage = (batch_advantage - np.mean(batch_advantage))/np.std(batch_advantage)
        batch_target = np.expand_dims(batch_target, 1)
        length = len(batch_state)
        actor_loss = 0
        critic_loss = 0
        random_sequence = np.random.choice(length, self.batch_size*self.epochs, replace=False)
        minibatch_state = []
        minibatch_action = []
        minibatch_target = []
        minibatch_advantage = []
        minibatch_log_probs = []
        for k in random_sequence:
            minibatch_state.append(batch_state[k])
            minibatch_action.append(batch_action[k])
            minibatch_target.append(batch_target[k])
            minibatch_advantage.append(batch_advantage[k])
            minibatch_log_probs.append(batch_log_probs[k])
            if len(minibatch_state) == self.batch_size:
                actor_loss += self.actor.learn(minibatch_state, minibatch_action,
                                               minibatch_advantage, minibatch_log_probs)
                critic_loss += self.critic.learn(minibatch_state, minibatch_target)
                minibatch_state = []
                minibatch_action = []
                minibatch_target = []
                minibatch_advantage = []
                minibatch_log_probs = []

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
            query_intervals = 2
            last_interval = 0
            max_score = -20000
            best_validation_score = -500
            for games in range(n_games):
                state = self.env.reset()
                score = 0
                while True:
                    if self.doScale:
                        state = self.scale(state)
                    action, log_probs = self.act(state)
                    state_, reward, done, _ = self.env.step(action)
                    score += reward
                    # print(reward)
                    # self.env.render()
                    if np.isnan(reward):
                        print("ERROR")
                        sys.exit()
                    if self.doScale:
                        self.remember(state, action, reward, self.scale(state_), done, log_probs)
                    else:
                        self.remember(state, action, reward, state_, done, log_probs)
                    # print(self.memory.size())
                    state = state_
                    if done:
                        break
                score_history += score
                max_score = max(score, max_score)
                last_interval += 1
                # print(self.memory.size())
                if self.memory.size() > self.memory_buffer_length:
                    actor_loss_, critic_loss_ = self.learn()
                    actor_losses += actor_loss_
                    critic_losses += critic_loss_
                    if last_interval >= query_intervals:
                        validation_score, _ = self.test_play(False, 10)
                        print("Games: {}, Score: {:.2f}, Max Score: {:.2f}, Validation: {:.2f},"
                              .format(games+1, score_history/last_interval,
                                      max_score, validation_score),
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--scale", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dir", type=str, default="chkpt")
    settings = parser.parse_args()
    if not settings.env.find("Bullet") == -1:
        import pybullet_envs

    if not settings.test:
        env = gym.make(settings.env)
    else:
        if settings.env.find("Bullet") == -1:
            env = gym.make(settings.env)
        else:
            env = gym.make(settings.env, render=True)
    agent = PPOAgent(env, settings.resume, settings.scale, dir=settings.dir)
    try:
        agent.play(settings.test, True)
    except KeyboardInterrupt:
        inp = input("Wanna save model? Press Y.\n")
        if inp == 'Y' or inp == 'y' and not settings.test:
            agent.save_checkpoint()
    finally:
        print("okbye!")
