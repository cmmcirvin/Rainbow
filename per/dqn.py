import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, num_states, num_actions):
        super().__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        self.gamma = 0.99

        self.layers = nn.Sequential(
            nn.Linear(self.num_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

    def forward(self, x):
        return self.layers(x)
