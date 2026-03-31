"""PPO algorithm implementation.

Supports vectorized environments and GAE advantage estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    clip_eps: float = 0.2
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    mini_batch_size: int = 64
    n_steps: int = 2048
    n_envs: int = 8
    total_timesteps: int = 2_000_000
    action_scale: float = 50.0  # scale from [-1,1] to [-50,50]


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: np.ndarray, gamma: float, lam: float) -> None:
        """Generalized Advantage Estimation."""
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            delta = (
                self.rewards[t]
                + gamma * next_value * (1 - self.dones[t])
                - self.values[t]
            )
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield mini-batches for PPO update."""
        total = self.n_steps * self.n_envs
        indices = np.arange(total)
        np.random.shuffle(indices)

        obs_flat = self.obs.reshape(total, -1)
        act_flat = self.actions.reshape(total, -1)
        logp_flat = self.log_probs.reshape(total)
        adv_flat = self.advantages.reshape(total)
        ret_flat = self.returns.reshape(total)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield (
                torch.from_numpy(obs_flat[idx]),
                torch.from_numpy(act_flat[idx]),
                torch.from_numpy(logp_flat[idx]),
                torch.from_numpy(adv_flat[idx]),
                torch.from_numpy(ret_flat[idx]),
            )

    def reset(self) -> None:
        self.ptr = 0


def ppo_update(
    policy: nn.Module,
    value_net: nn.Module,
    buffer: RolloutBuffer,
    config: PPOConfig,
    opt_policy: torch.optim.Optimizer,
    opt_value: torch.optim.Optimizer,
    device: str = "cuda",
) -> dict[str, float]:
    """Perform PPO update on collected rollout data.

    Returns dict of logged metrics.
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(config.n_epochs):
        for obs, act, old_logp, adv, ret in buffer.get_batches(config.mini_batch_size):
            obs = obs.to(device)
            act = act.to(device)
            old_logp = old_logp.to(device)
            adv = adv.to(device)
            ret = ret.to(device)

            # Policy loss
            dist = policy.get_dist(obs)
            new_logp = dist.log_prob(act).sum(-1)
            ratio = (new_logp - old_logp).exp()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().sum(-1).mean()

            # Value loss
            value_pred = value_net(obs)
            value_loss = nn.functional.mse_loss(value_pred, ret)

            # Combined update (separate optimizers but can be done together)
            loss_pi = policy_loss - config.entropy_coef * entropy
            opt_policy.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            opt_policy.step()

            loss_v = config.value_coef * value_loss
            opt_value.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), config.max_grad_norm)
            opt_value.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }
