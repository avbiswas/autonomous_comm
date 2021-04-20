import numpy as np


class Memory():

    def __init__(self):
        self.memory = {}
        self.memory['state'] = []
        self.memory['action'] = []
        self.memory['reward'] = []
        self.memory['state_next'] = []
        self.memory['done'] = []
        self.memory['old_logprobs'] = []
        self.memory['time_limit_reached'] = []

    def size(self):
        return len(self.memory['state'])

    def remember(self, state, action, reward, state_next, done, old_logprobs,
                 time_limit_reached=None):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['reward'].append(reward)
        self.memory['state_next'].append(state_next)
        self.memory['done'].append(done)
        self.memory['old_logprobs'].append(old_logprobs)
        if time_limit_reached is None:
            self.memory['time_limit_reached'].append(done)
        else:
            self.memory['time_limit_reached'].append(time_limit_reached)

    def getRecords(self):
        return self.memory['state'], self.memory['action'], self.memory['reward'], \
            self.memory['state_next'], self.memory['done'], self.memory['old_logprobs'], \
            self.memory['time_limit_reached']

    def clear(self):
        self.memory = {}
        self.memory['state'] = []
        self.memory['action'] = []
        self.memory['reward'] = []
        self.memory['state_next'] = []
        self.memory['done'] = []
        self.memory['old_logprobs'] = []
        self.memory['time_limit_reached'] = []


class VectorizedMemory():
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.memory_envs = {i: Memory() for i in range(self.num_envs)}

    def size(self):
        return sum([memory.size() for _, memory in self.memory_envs.items()])

    def remember(self, states, actions, rewards, states_next, dones, old_logprobs,
                 time_limit_reached=None):
        # print(states.keys(), rewards.keys(), dones, states_next.keys())
        if time_limit_reached is None:
            time_limit_reached = dones
        for key in states_next:
            self.memory_envs[key].remember(states[key],
                                           actions[key],
                                           rewards[key],
                                           states_next[key],
                                           dones[key],
                                           old_logprobs[key],
                                           time_limit_reached[key])

    def getRecords(self):
        states = []
        actions = []
        rewards = []
        states_next = []
        dones = []
        old_logprobs = []
        time_limit_reached = []
        for _ in range(self.num_envs):
            s, a, r, s_, d, lp, tlr = self.memory_envs[_].getRecords()
            states.extend(s)
            actions.extend(a)
            rewards.extend(r)
            states_next.extend(s_)
            dones.extend(d)
            old_logprobs.extend(lp)
            time_limit_reached.extend(tlr)
        return states, actions, rewards, states_next, dones, old_logprobs, time_limit_reached

    def clear(self):
        [self.memory_envs[k].clear() for k in range(self.num_envs)]


if __name__ == '__main__':
    memory = Memory()
    memory.remember((1, 2), 1, 0, (2, 2), False)
    memory.remember((2, 2), 2, 1, (3, 2), False)
    memory.remember((3, 2), 1, 0, (2, 3), True)
    memory.remember((2, 3), 2, 0, (3, 3), False)
    sampled_state, sampled_action, sampled_reward, sampled_state_next, sampled_done = \
        memory.sample(3)
    print("State", sampled_state)
    print("Action", sampled_action)
    print("Reward", sampled_reward)
    print("Next", sampled_state_next)
    print("Done", sampled_done)
