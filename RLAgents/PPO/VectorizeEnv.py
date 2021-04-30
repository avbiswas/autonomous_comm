import gym
import highway_env
import numpy as np
from .utils import tile_images
from .memory_GAE import VectorizedMemory
import time


class VectorizedEnvs:
    def __init__(self, env_fn, num_envs, observation="Kinematics"):
        self.num_envs = num_envs
        self.envs = []
        for _ in range(self.num_envs):
            self.envs.append(env_fn())
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.dones = np.zeros(self.num_envs)

    def reset(self):
        obs = {}
        for i in range(self.num_envs):
            obs[i] = np.array(self.envs[i].reset())
        self.dones = np.zeros(self.num_envs)
        return obs

    def step(self, actions):
        obs_ = {}
        rewards = {}
        dones = {}
        infos = {}
        for i in range(self.num_envs):
            if self.dones[i]:
                continue
            obs, r, d, info = self.envs[i].step(actions[i])
            if d:
                self.dones[i] = 1
            obs_[i] = np.array(obs)
            rewards[i] = r
            dones[i] = d
            infos[i] = info
        return obs_, rewards, dones, infos

    def get_images(self):
        def add_border(img):
            h, w, c = img.shape
            padding = [10, 10]
            bordered_image = np.zeros([h + padding[0]*2, w + padding[1] * 2, c], dtype=np.uint8)
            bordered_image[padding[0]:-padding[0], padding[1]:-padding[1], :] = img
            return bordered_image

        return [add_border(env.render(mode='rgb_array')) for env in self.envs]

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(2)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def sample_action(self):
        return {i: self.envs[i].action_space.sample() for i in range(self.num_envs)}

    def close(self):
        for i in range(self.num_envs):
            self.envs[i].close()


if __name__ == "__init__":
    num_envs = 8
    start_time = time.time()
    env = VectorizedEnvs(num_envs)
    memory = VectorizedMemory(num_envs)

    x = 0
    while True:
        s = env.reset()
        while True:
            a = env.sample_action()
            s_, r, d, _ = env.step(a)
            memory.remember(s, a, r, s_, d, a)
            s = s_
            # env.render()
            dones = [v for _, v in d.items()]
            x += len(dones)
            if np.all(dones):
                break
        # print(memory.size())
        if memory.size() > 64:
            break

    end_time = time.time()
    print(end_time - start_time)

    s, a, r, s_, d, lp = memory.getRecords()
    s = np.array(s)
    a = np.array(a)
    r = np.array(r)
    d = np.array(d)
    lp = np.array(lp)
    # print(s)
    print(np.shape(s))
    print(np.shape(a))
    print(np.shape(r))
    print(np.shape(s_))
    print(d)
    # print(lp)
