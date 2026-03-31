"""Policy and value networks for PPO variants.

Five architectures:
1. VanillaPolicy: standard MLP(20->256->256->7)
2. GeometricPolicy: 2-head split {j1-j4} + {j5-j7}
3. CouplingAwarePolicy: MLP(41->256->256->7) with 21-dim structure features
4. QuantumDecomposedPolicy: multi-head policy with entanglement-based grouping
5. ValueNet: standard value function MLP
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class VanillaPolicy(nn.Module):
    """Standard MLP policy: obs(20) -> mean(7), logstd(7)."""

    def __init__(self, obs_dim: int = 20, act_dim: int = 7, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class GeometricPolicy(nn.Module):
    """Two-head policy: proximal(j1-j4) + distal(j5-j7).

    Obs split: shared features, but separate action heads.
    Motivated by the structural prior: proximal joints dominate dynamics.
    """

    def __init__(self, obs_dim: int = 20, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Proximal: joints 1-4 (indices 0-3)
        self.prox_head = nn.Linear(hidden, 4)
        self.prox_log_std = nn.Parameter(torch.zeros(4))
        # Distal: joints 5-7 (indices 4-6)
        self.dist_head = nn.Linear(hidden, 3)
        self.dist_log_std = nn.Parameter(torch.zeros(3))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        mean_prox = self.prox_head(h)
        mean_dist = self.dist_head(h)
        mean = torch.cat([mean_prox, mean_dist], dim=-1)
        std = torch.cat([
            self.prox_log_std.exp().expand_as(mean_prox),
            self.dist_log_std.exp().expand_as(mean_dist),
        ], dim=-1)
        return mean, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class CouplingAwarePolicy(nn.Module):
    """Policy with structure-feature augmented observations.

    obs_dim=41: [q(7), dq(7), ee_pos(3), target_pos(3)] + structure_features(21).
    Structure features are either |J_ij| (classical) or C_ij (quantum).
    Same architecture for both -> fair comparison.
    """

    def __init__(self, obs_dim: int = 41, act_dim: int = 7, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class QuantumDecomposedPolicy(nn.Module):
    """Multi-head policy with entanglement-based joint grouping.

    obs_dim=41: base(20) + structure_features(21).
    Joint groups determined by entanglement spectral clustering.
    Default grouping at q=0: set at init, updated via set_groups().

    vs GeometricPolicy (fixed {0-3}+{4-6}): this adapts to dynamics.
    """

    def __init__(
        self,
        obs_dim: int = 41,
        act_dim: int = 7,
        hidden: int = 256,
        default_groups: list[list[int]] | None = None,
    ):
        super().__init__()
        self.act_dim = act_dim
        if default_groups is None:
            # Default: same as GeometricPolicy until first set_groups call
            default_groups = [[0, 1, 2, 3], [4, 5, 6]]
        self._groups = default_groups

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Create heads for max possible groups (up to act_dim singletons)
        # We use a ModuleDict keyed by group size for flexibility
        self._heads = nn.ModuleList()
        self._head_log_stds = nn.ParameterList()
        for group in default_groups:
            self._heads.append(nn.Linear(hidden, len(group)))
            self._head_log_stds.append(nn.Parameter(torch.zeros(len(group))))

    def set_groups(self, groups: list[list[int]]) -> None:
        """Update joint grouping (e.g., from new entanglement clustering).

        Note: head dimensions must match. If group sizes change,
        we reinitialize the affected heads.
        """
        if len(groups) != len(self._groups):
            # Rebuild heads (rare: only when cluster count changes)
            device = next(self.parameters()).device
            hidden = self.shared[-2].out_features
            self._heads = nn.ModuleList()
            self._head_log_stds = nn.ParameterList()
            for group in groups:
                self._heads.append(nn.Linear(hidden, len(group)).to(device))
                self._head_log_stds.append(
                    nn.Parameter(torch.zeros(len(group), device=device))
                )
        self._groups = groups

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        batch_size = obs.shape[0] if obs.dim() > 1 else 1

        mean = torch.zeros(batch_size, self.act_dim, device=obs.device)
        std = torch.zeros(batch_size, self.act_dim, device=obs.device)

        for group, head, log_std in zip(
            self._groups, self._heads, self._head_log_stds
        ):
            group_mean = head(h)
            group_std = log_std.exp().expand_as(group_mean)
            for k, joint_idx in enumerate(group):
                mean[..., joint_idx] = group_mean[..., k]
                std[..., joint_idx] = group_std[..., k]

        if obs.dim() == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        return mean, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class ValueNet(nn.Module):
    """Value function: obs -> V(scalar)."""

    def __init__(self, obs_dim: int = 20, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
