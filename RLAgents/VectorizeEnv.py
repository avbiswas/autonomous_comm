import gym
import highway_env
import numpy as np
from .utils import tile_images
from .memory_GAE import VectorizedMemory
import time


class VectorizedEnvs:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = []
        for _ in range(self.num_envs):
            env = gym.make('highway-v0')
            env.configure({
                "action": {
                    "type": "ContinuousAction"
                },
                "offroad_terminal": True,
                "simulation_frequency": 8,
                "policy_frequency": 4,
                "offscreen_rendering": True
            })
            env.reset()
            self.envs.append(env)
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
        return [env.render(mode='rgb_array') for env in self.envs]

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
