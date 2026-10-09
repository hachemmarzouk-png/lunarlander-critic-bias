"""Minimal, inspectable DDPG / TD3 implementations.

- Two-timescale targets are Polyak-averaged copies.
- TD3 adds clipped double-Q, target-policy smoothing and delayed actor steps.
- DDPG uses one critic with no policy smoothing.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .networks import Actor, Critic


@dataclass
class AgentConfig:
    algorithm: str = "td3"
    layer_norm: bool = False
    ln_where: str = "both"  # both, actor or critic (only used when layer_norm=True)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    hidden: tuple[int, int] = (256, 256)
    device: str = "cpu"


class Agent:
    def __init__(self, obs_dim: int, act_dim: int, max_action: float,
                 cfg: AgentConfig):
        if cfg.algorithm not in ("ddpg", "td3"):
            raise ValueError("algorithm must be ddpg or td3")
        if cfg.ln_where not in ("both", "actor", "critic"):
            raise ValueError("ln_where must be both, actor or critic")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.max_action = float(max_action)
        actor_ln = cfg.layer_norm and cfg.ln_where in ("both", "actor")
        critic_ln = cfg.layer_norm and cfg.ln_where in ("both", "critic")
        self.actor = Actor(obs_dim, act_dim, max_action, actor_ln, cfg.hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1 = Critic(obs_dim, act_dim, critic_ln, cfg.hidden).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2 = (Critic(obs_dim, act_dim, critic_ln, cfg.hidden).to(self.device)
                        if cfg.algorithm == "td3" else None)
        self.critic2_target = copy.deepcopy(self.critic2) if self.critic2 is not None else None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        params = list(self.critic1.parameters())
        if self.critic2 is not None:
            params += list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(params, lr=cfg.critic_lr)
        self.gradient_steps = 0

    @torch.no_grad()
    def act(self, state: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(tensor).cpu().numpy()[0]

    @torch.no_grad()
    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        q1 = self.critic1(s, a).squeeze(-1).cpu().numpy()
        values = {"q1": q1}
        if self.critic2 is not None:
            q2 = self.critic2(s, a).squeeze(-1).cpu().numpy()
            values.update(q2=q2, qmin=np.minimum(q1, q2))
        return values

    @torch.no_grad()
    def bellman_target(self, reward: torch.Tensor, next_state: torch.Tensor,
                       terminated: torch.Tensor) -> torch.Tensor:
        next_action = self.actor_target(next_state)
        if self.critic2_target is not None:
            noise = (torch.randn_like(next_action) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            next_q = torch.minimum(self.critic1_target(next_state, next_action),
                                   self.critic2_target(next_state, next_action))
        else:
            next_q = self.critic1_target(next_state, next_action)
        return reward + self.cfg.gamma * (1.0 - terminated) * next_q

    @staticmethod
    @torch.no_grad()
    def _polyak(source: nn.Module, target: nn.Module, tau: float):
        for source_p, target_p in zip(source.parameters(), target.parameters()):
            target_p.mul_(1 - tau).add_(source_p, alpha=tau)
        # LayerNorm has only parameters; no running statistics to update.

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        s, a, r, ns, done = batch
        cfg = self.cfg
        target = self.bellman_target(r, ns, done)
        q1 = self.critic1(s, a)
        loss = (q1 - target).square().mean()
        if self.critic2 is not None:
            loss = loss + (self.critic2(s, a) - target).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self.gradient_steps += 1
        update_actor = cfg.algorithm == "ddpg" or self.gradient_steps % cfg.policy_delay == 0
        output = {"critic_loss": float(loss.detach().item()), "actor_updated": float(update_actor)}
        if update_actor:
            # Critic is fixed while optimizing actor, though backprop passes through its input.
            for p in self.critic1.parameters():
                p.requires_grad_(False)
            actor_loss = -self.critic1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for p in self.critic1.parameters():
                p.requires_grad_(True)
            self._polyak(self.actor, self.actor_target, cfg.tau)
            self._polyak(self.critic1, self.critic1_target, cfg.tau)
            if self.critic2 is not None:
                self._polyak(self.critic2, self.critic2_target, cfg.tau)
            output["actor_loss"] = float(actor_loss.detach().item())
        return output

    def save(self, path: str):
        torch.save({
            "config": vars(self.cfg), "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2": None if self.critic2 is None else self.critic2.state_dict(),
            "critic2_target": None if self.critic2_target is None else self.critic2_target.state_dict(),
            "gradient_steps": self.gradient_steps,
        }, path)
