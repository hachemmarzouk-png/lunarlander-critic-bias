"""Actor and critic multilayer perceptrons with optional LayerNorm.

LayerNorm is inserted after each *hidden* linear layer and before ReLU.
Its parameters are learned. Outputs are never normalized.
"""
from __future__ import annotations
import torch
from torch import nn


def mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int,
        layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for h in hidden:
        layers.append(nn.Linear(current, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        current = h
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_action: float = 1.,
                 layer_norm: bool = False, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, act_dim, layer_norm)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.net(state))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, layer_norm: bool = False,
                 hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden, 1, layer_norm)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))
