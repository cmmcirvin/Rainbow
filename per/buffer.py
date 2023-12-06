import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class Buffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def store(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedBuffer:

    def __init__(self, capacity, alpha, beta):
        self.alpha = alpha
        self.beta = beta
#        self.memory = deque([], maxlen=capacity)
#        self.priorities = deque([], maxlen=capacity)
        self.capacity = capacity
        self.size = 0
        self.curr_idx = 0

        self.memory = np.zeros(capacity, dtype=object)
        self.priorities = np.zeros(capacity)

    def store(self, priority, *args):
#        self.memory.append(Transition(*args))
#        self.priorities.append(priority)

        self.memory[self.curr_idx] = Transition(*args)
        self.priorities[self.curr_idx] = priority

        self.size = min(self.size + 1, self.capacity)
        self.curr_idx = (self.curr_idx + 1) % self.capacity

    def max_priority(self):
#        if len(self) == 0:
        if self.size == 0:
            return 1.0
        return max(self.priorities)

    def importance_weights(self, priorities):
#        N = len(self.memory)
        N = self.size
        max_weight = (N * self.max_priority()) ** (-self.beta)
        weights = (N * priorities) ** (-self.beta) / max_weight
        return weights

    def update_priorities(self, priorities, idxes):
        self.priorities[idxes] = priorities
#        i = 0
#        for idx in idxes:
#            self.priorities[idx] = priorities[i]
#            i += 1

    def sample(self, batch_size):

#        priorities = np.array(self.priorities)
#        transitions = np.array(self.memory, dtype=object)
#        probs = priorities / sum(priorities)
        probs = self.priorities[:self.size] / np.sum(self.priorities[:self.size])

        sample_idxes = np.random.choice(np.arange(0, self.size), size=batch_size, p=probs)

#        return sample_idxes, priorities[sample_idxes], transitions[sample_idxes]
        return sample_idxes, self.priorities[sample_idxes], self.memory[sample_idxes]


    def __len__(self):
#        return len(self.memory)
        return self.size
