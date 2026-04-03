"""Modal action and coupling-features policy networks for toy dual-arm.

ModalActionPolicy:  outputs [tau_within_L(2), tau_within_R(2), a_modal(2)]
                    mapped to 4-dim torques via SVD of cross-arm coupling.
CouplingFeaturesPolicy: standard MLP with augmented obs (base + J_cross flat).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


class ModalActionPolicy(nn.Module):
    """Policy that decomposes actions into within-arm + cross-arm modal.

    Raw output (6-dim): [tau_within_L(2), tau_within_R(2), a_modal(2)].
    Call ``map_to_torques`` to convert to physical 4-dim joint torques.
    """

    RAW_DIM = 6  # 2 + 2 + 2

    def __init__(self, obs_dim: int = 13, n_per_arm: int = 2, hidden: int = 128):
        super().__init__()
        self.n_per_arm = n_per_arm
        raw_dim = 3 * n_per_arm  # 6
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, raw_dim)
        self.log_std = nn.Parameter(torch.zeros(raw_dim))

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        h = self.shared(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, obs: Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def get_action(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    @staticmethod
    def map_to_torques(
        raw: Tensor,
        U: Tensor,
        sigma: Tensor,
        Vh: Tensor,
        action_scale: float = 10.0,
    ) -> Tensor:
        """Map 6-dim raw action to 4-dim physical torques.

        Args:
            raw: (batch, 6) policy output
            U: (batch, 2, 2) left singular vectors of J_cross
            sigma: (batch, 2) singular values
            Vh: (batch, 2, 2) right singular vectors
            action_scale: torque scale factor

        Returns:
            (batch, 4) physical joint torques
        """
        tau_within_L = raw[:, :2] * action_scale
        tau_within_R = raw[:, 2:4] * action_scale
        a_modal = raw[:, 4:] * action_scale

        # Weighted modal coefficients: sigma_i * a_modal_i
        weighted = sigma * a_modal  # (batch, 2)

        # Project onto left/right singular vectors
        tau_cross_L = torch.bmm(U, weighted.unsqueeze(-1)).squeeze(-1)   # (batch, 2)
        V = Vh.transpose(-1, -2)
        tau_cross_R = torch.bmm(V, weighted.unsqueeze(-1)).squeeze(-1)   # (batch, 2)

        tau_L = tau_within_L + tau_cross_L
        tau_R = tau_within_R + tau_cross_R
        return torch.cat([tau_L, tau_R], dim=-1)


class CouplingFeaturesPolicy(nn.Module):
    """Standard MLP policy with coupling features appended to observation.

    obs_dim = 13 (base) + 4 (J_cross flat) = 17.
    """

    def __init__(self, obs_dim: int = 17, act_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        h = self.shared(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, obs: Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def get_action(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
