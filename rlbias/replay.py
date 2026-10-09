"""Uniform off-policy replay; time-limit truncation does not mark a true terminal."""
from __future__ import annotations
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.size, self.index = 0, 0

    def add(self, state, action, reward, next_state, terminated: bool):
        i = self.index
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i, 0] = reward
        self.next_states[i] = next_state
        self.terminated[i, 0] = float(terminated)
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, device: str = "cpu"):
        idx = self.rng.integers(0, self.size, size=n)
        return tuple(torch.as_tensor(x[idx], dtype=torch.float32, device=device)
                     for x in (self.states, self.actions, self.rewards,
                               self.next_states, self.terminated))
