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
        # self.memory['old_sigma'] = []

    def size(self):
        return len(self.memory['state'])

    def remember(self, state, action, reward, state_next, done, old_logprobs):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['reward'].append(reward)
        self.memory['state_next'].append(state_next)
        self.memory['done'].append(done)
        self.memory['old_logprobs'].append(old_logprobs)

    def sample(self, batch_size):
        length = len(self.memory["state"])
        if (batch_size >= length):
            return self.memory['state'], self.memory['action'], self.memory['reward'], \
                self.memory['state_next'], self.memory['done']

        random_sequence = np.random.choice(length, batch_size, replace=False)
        sampled_state = []
        sampled_action = []
        sampled_reward = []
        sampled_state_next = []
        sampled_done = []
        for i in random_sequence:
            sampled_state.append(self.memory['state'][i])
            sampled_action.append(self.memory['action'][i])
            sampled_reward.append(self.memory['reward'][i])
            sampled_state_next.append(self.memory['state_next'][i])
            sampled_done.append(self.memory['done'][i])
        return sampled_state, sampled_action, sampled_reward, sampled_state_next, sampled_done

    def getRecords(self):
        return self.memory['state'], self.memory['action'], self.memory['reward'], \
            self.memory['state_next'], self.memory['done'], self.memory['old_logprobs']

    def clear(self):
        self.memory = {}
        self.memory['state'] = []
        self.memory['action'] = []
        self.memory['reward'] = []
        self.memory['state_next'] = []
        self.memory['done'] = []
        self.memory['old_logprobs'] = []


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
