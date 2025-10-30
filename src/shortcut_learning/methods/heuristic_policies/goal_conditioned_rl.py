"""Basic implementation of goal-conditioned reinforcement learning.

Definitely requires more advanced techniques!
"""

from __future__ import annotations

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from shortcut_learning.methods.graph_utils import PlanningGraphNode
from shortcut_learning.methods.wrappers import SLAPWrapperV2

class GoalConditionedWrapper(gym.Wrapper):
    """Wrapper to make any gym environment goal-conditioned with -1 step
    reward."""

    def __init__(self, env: SLAPWrapperV2):
        super().__init__(env)
        self.goal = 

        # Modify observation space to include goal
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, env.observation_space.low]),
            high=np.concatenate(
                [env.observation_space.high, env.observation_space.high]
            ),
            dtype=np.float32,
        )

    def set_goal(self, goal):
        """Set the goal state for this episode."""
        self.goal = np.array(goal, dtype=np.float32)

    def _get_obs(self, state):
        """Concatenate state and goal."""
        return np.concatenate([state, self.goal])

    def _reached_goal(self, state):
        """Compute distance between state and goal."""
        return state in set(self.goal.states)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        # If no goal is set, sample a random goal from observation space
        if self.goal is None:
            self.goal = self.env.observation_space.sample()

        return self._get_obs(state), info

    def step(self, action):
        state, _, terminated, truncated, info = self.env.step(action)

        # Check if we've reached the goal
        goal_reached = self._reached_goal(state)

        # Reward is -1 for each step, 0 when goal is reached
        reward = 0.0 if goal_reached else -1.0

        # Episode ends when goal is reached
        terminated = terminated or goal_reached

        info["goal_reached"] = goal_reached

        return self._get_obs(state), reward, terminated, truncated, info


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random experience from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-network for goal-conditioned Q-learning."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)


class GoalConditionedDQN:
    """DQN agent for goal-conditioned learning."""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=10,
        device="cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_count = 0

    def select_action(self, state, behavior_policy="epsilon_greedy"):
        """Select action using behavior policy."""
        if behavior_policy == "epsilon_greedy" and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update(self):
        """Update Q-network using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_min_path_distance(self, state, goal):
        """Get the minimum action path distance from state to goal.

        This is the negative of the Q-value for the best action.
        """
        state_goal = np.concatenate([state, goal])
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_goal).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            # Q-value represents negative path length (since reward is -1 per step)
            return -q_values.max().item()


def train_goal_conditioned_rl(
    env, agent, num_episodes=1000, max_steps=200, verbose=True
):
    """Train the goal-conditioned agent."""
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Update agent
            _ = agent.update()

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return episode_rewards, episode_lengths
