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
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)

    def store(self, priority, *args):
#        self.memory.append((priority, Transition(*args)))
        self.memory.append(Transition(*args))
        self.priorities.append(priority)


    def max_priority(self):
        if len(self) == 0:
            return 1.0
        return max(self.priorities)

    def importance_weights(self, priorities):
        N = len(self.memory)
        max_weight = (N * self.max_priority()) ** (-self.beta)
        weights = (N * priorities) ** (-self.beta) / max_weight
        return weights

    def update_priorities(self, priorities, idxes):
        i = 0
        for idx in idxes:
            self.priorities[idx] = priorities[i]
            i += 1

    def sample(self, batch_size):

        priorities = np.array(self.priorities)
        transitions = np.array(self.memory, dtype=object)
        probs = priorities / sum(priorities)

        sample_idxes = np.random.choice(np.arange(0, len(self.memory)), size=batch_size, p=probs)

        return sample_idxes, priorities[sample_idxes], transitions[sample_idxes]


    def __len__(self):
        return len(self.memory)