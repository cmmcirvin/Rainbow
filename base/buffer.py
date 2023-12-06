from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class Buffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.curr_idx = 0
        self.memory = np.zeros(capacity, dtype=object)

    def store(self, *args):
        self.memory[self.curr_idx] = Transition(*args)
        self.size = min(self.size + 1, self.capacity)
        self.curr_idx = (self.curr_idx + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(self.memory[:self.size], size=batch_size)

    def __len__(self):
        return self.size
