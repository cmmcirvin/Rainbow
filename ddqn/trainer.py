import torch
import torch.nn as nn
import gymnasium as gym
import math
import matplotlib.pyplot as plt

from tqdm import tqdm
from dqn import DQN
from buffer import Buffer
from torch.utils.tensorboard.writer import SummaryWriter

def get_action(state, steps):
    epsilon = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps / EPS_DECAY)
    if torch.rand(1) < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        return agent(state).argmax().item()

def calculate_loss(batch):

    states, actions, rewards, next_states = zip(*batch)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in next_states if s is not None])
    state_batch = torch.stack(states)
    action_batch = torch.stack(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = agent(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size)
    with torch.no_grad():
        selected_actions = agent(non_final_next_states).argmax(dim=1, keepdim=True)
        next_state_values[non_final_mask] = target(non_final_next_states).gather(1, selected_actions).flatten()
    
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    return loss

def update_params():
    if len(buffer) < batch_size:
        return

    batch = buffer.sample(batch_size)
    loss = calculate_loss(batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    env = gym.make("CartPole-v1", max_episode_steps=200)
    writer = SummaryWriter()
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    tau = 0.005

    agent = DQN(num_states, num_actions)
    target = DQN(num_states, num_actions)
    target.load_state_dict(agent.state_dict())

    buffer = Buffer(10000)
    rewards = []

    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4, amsgrad=True)
    num_epochs = 1000
    batch_size = 128
    gamma = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000

    steps = 0

    pbar = tqdm(range(num_epochs))

    for epoch in pbar:

        state, _ = env.reset()
        state = torch.tensor(state)

        terminated = False
        truncated = False
        ep_reward = 0

        while not terminated and not truncated:
            with torch.no_grad():
                action = get_action(state, steps)
                steps += 1

            next_state, reward, terminated, truncated, _ = env.step(action)

            ep_reward += float(reward)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state)

            buffer.store(state, torch.tensor([action]), torch.tensor([reward]), next_state)
            state = next_state
            update_params()

            target_state_dict = target.state_dict()
            agent_state_dict = agent.state_dict()
            for key in agent_state_dict:
                target_state_dict[key] = agent_state_dict[key] * tau + target_state_dict[key]*(1-tau)
            target.load_state_dict(target_state_dict)

        writer.add_scalar("Reward", ep_reward, epoch)
        pbar.set_description(f"Reward: {ep_reward:.0f}")
        rewards.append(ep_reward)

    plt.plot(rewards)
    plt.show()
