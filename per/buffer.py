import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class PrioritizedBuffer:

    def __init__(self, capacity, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.size = 0
        self.curr_idx = 0
        self.memory = np.zeros(capacity, dtype=object)
        self.priorities = np.zeros(capacity)

    def store(self, priority, *args):
        self.memory[self.curr_idx] = Transition(*args)
        self.priorities[self.curr_idx] = priority
        self.size = min(self.size + 1, self.capacity)
        self.curr_idx = (self.curr_idx + 1) % self.capacity

    def max_priority(self):
        if self.size == 0:
            return 1.0
        return max(self.priorities)

    def importance_weights(self, priorities):
        N = self.size
        max_weight = (N * self.max_priority()) ** (-self.beta)
        weights = (N * priorities) ** (-self.beta) / max_weight
        return weights

    def update_priorities(self, priorities, idxes):
        self.priorities[idxes] = priorities

    def sample(self, batch_size):
        probs = self.priorities[:self.size] / np.sum(self.priorities[:self.size])
        sample_idxes = np.random.choice(np.arange(0, self.size), size=batch_size, p=probs)
        return sample_idxes, self.priorities[sample_idxes], self.memory[sample_idxes]


    def __len__(self):
        return self.size
