"""GPU-vectorized 2x2-DOF toy dual-arm environment.

Two planar double-pendulums sharing a common object.
M_eff = diag(M_L + J_L^T M_obj J_L, M_R + J_R^T M_obj J_R)
      + off-diag(J_L^T M_obj J_R, transpose)

Physics: semi-implicit Euler on GPU, no MuJoCo dependency.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ToyDualArmVecEnv:
    """GPU-vectorized 2x2-DOF dual-arm environment.

    Observation (13-dim): [q_L(2), q_R(2), dq_L(2), dq_R(2),
                           q_target_L(2), q_target_R(2), phase]
    Action (4-dim): joint torques (clamped to [-action_scale, action_scale])
    Reward: -||q - q_target||^2 - 0.01 * ||tau||^2
    """

    OBS_DIM = 13
    ACT_DIM = 4

    def __init__(
        self,
        n_envs: int = 512,
        n_per_arm: int = 2,
        object_mass: float = 0.5,
        link_masses: tuple[float, float] = (1.0, 0.5),
        link_lengths: tuple[float, float] = (0.3, 0.25),
        damping: float = 0.5,
        dt: float = 0.02,
        max_steps: int = 200,
        traj_freq_range: tuple[float, float] = (0.5, 0.8),
        traj_amp_range: tuple[float, float] = (0.3, 0.6),
        action_scale: float = 10.0,
        device: str = "cuda",
    ):
        self.n_envs = n_envs
        self.n_per_arm = n_per_arm
        self.n_dof = 2 * n_per_arm
        self.device = device
        self.dt = dt
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.damping = damping
        self.traj_freq_range = traj_freq_range
        self.traj_amp_range = traj_amp_range

        # Link parameters
        m1, m2 = link_masses
        l1, l2 = link_lengths
        self.l1 = l1
        self.l2 = l2
        lc1, lc2 = l1 / 2, l2 / 2
        I1 = m1 * l1**2 / 12
        I2 = m2 * l2**2 / 12

        # Precomputed constants for 2-DOF planar arm mass matrix:
        #   M = [[alpha + 2*beta*cos(q2), delta + beta*cos(q2)],
        #        [delta + beta*cos(q2),   delta              ]]
        self.alpha = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2)
        self.beta = m2 * l1 * lc2
        self.delta = I2 + m2 * lc2**2

        # Object mass matrix (2D planar translational: m_obj * I_2)
        self.object_mass = object_mass
        self.M_obj = object_mass * torch.eye(2, device=device).unsqueeze(0)  # (1,2,2)

        # State tensors
        self.q = torch.zeros(n_envs, self.n_dof, device=device)
        self.dq = torch.zeros(n_envs, self.n_dof, device=device)
        self.step_count = torch.zeros(n_envs, dtype=torch.long, device=device)

        # Target trajectory parameters (per env, per joint)
        self.q0 = torch.zeros(n_envs, self.n_dof, device=device)
        self.amp = torch.zeros(n_envs, self.n_dof, device=device)
        self.freq = torch.zeros(n_envs, self.n_dof, device=device)
        self.phi = torch.zeros(n_envs, self.n_dof, device=device)

        self.current_obs: Tensor | None = None

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _arm_mass_matrix(self, q_arm: Tensor) -> Tensor:
        """Single 2-DOF planar arm mass matrix. (batch,2) -> (batch,2,2)."""
        c2 = torch.cos(q_arm[:, 1])
        a = self.alpha + 2 * self.beta * c2
        b = self.delta + self.beta * c2
        d = self.delta
        M = torch.zeros(q_arm.shape[0], 2, 2, device=self.device)
        M[:, 0, 0] = a
        M[:, 0, 1] = b
        M[:, 1, 0] = b
        M[:, 1, 1] = d
        return M

    def _ee_jacobian(self, q_arm: Tensor) -> Tensor:
        """End-effector Jacobian for 2-DOF planar arm. (batch,2) -> (batch,2,2).

        EE position:
            x = l1*cos(q1) + l2*cos(q1+q2)
            y = l1*sin(q1) + l2*sin(q1+q2)
        """
        q1 = q_arm[:, 0]
        q12 = q1 + q_arm[:, 1]
        s1, c1 = torch.sin(q1), torch.cos(q1)
        s12, c12 = torch.sin(q12), torch.cos(q12)
        J = torch.zeros(q_arm.shape[0], 2, 2, device=self.device)
        J[:, 0, 0] = -self.l1 * s1 - self.l2 * s12
        J[:, 0, 1] = -self.l2 * s12
        J[:, 1, 0] = self.l1 * c1 + self.l2 * c12
        J[:, 1, 1] = self.l2 * c12
        return J

    def _compute_mass_matrix_batch(self, q: Tensor) -> Tensor:
        """Full 4x4 effective mass matrix with cross-arm coupling.

        q: (batch,4) -> M_eff: (batch,4,4) SPD.
        M_eff = diag(M_L + J_L^T M_obj J_L, M_R + J_R^T M_obj J_R)
              + off-diag(J_L^T M_obj J_R, transpose)
        """
        batch = q.shape[0]
        q_L, q_R = q[:, :2], q[:, 2:]
        M_L = self._arm_mass_matrix(q_L)
        M_R = self._arm_mass_matrix(q_R)
        J_L = self._ee_jacobian(q_L)
        J_R = self._ee_jacobian(q_R)
        M_obj = self.M_obj.expand(batch, -1, -1)

        JLt = J_L.transpose(-1, -2)
        JRt = J_R.transpose(-1, -2)
        cross = torch.bmm(torch.bmm(JLt, M_obj), J_R)
        aug_L = torch.bmm(torch.bmm(JLt, M_obj), J_L)
        aug_R = torch.bmm(torch.bmm(JRt, M_obj), J_R)

        M = torch.zeros(batch, 4, 4, device=self.device)
        M[:, :2, :2] = M_L + aug_L
        M[:, 2:, 2:] = M_R + aug_R
        M[:, :2, 2:] = cross
        M[:, 2:, :2] = cross.transpose(-1, -2)
        return M

    # ------------------------------------------------------------------
    # Target trajectory
    # ------------------------------------------------------------------

    def _generate_trajectory_params(self, mask: Tensor | None = None) -> None:
        """Randomize sinusoidal target: q_target(t) = q0 + A*sin(2*pi*f*t + phi)."""
        if mask is None:
            n = self.n_envs
            idx = slice(None)
        else:
            n = int(mask.sum().item())
            if n == 0:
                return
            idx = mask

        self.q0[idx] = torch.randn(n, self.n_dof, device=self.device) * 0.3
        flo, fhi = self.traj_freq_range
        self.freq[idx] = torch.rand(n, self.n_dof, device=self.device) * (fhi - flo) + flo
        alo, ahi = self.traj_amp_range
        self.amp[idx] = torch.rand(n, self.n_dof, device=self.device) * (ahi - alo) + alo
        self.phi[idx] = torch.rand(n, self.n_dof, device=self.device) * 2 * torch.pi

    def _get_target(self) -> Tensor:
        """Current target joint positions. (n_envs, 4)."""
        t = (self.step_count.float() * self.dt).unsqueeze(-1)
        return self.q0 + self.amp * torch.sin(2 * torch.pi * self.freq * t + self.phi)

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def _build_obs(self) -> Tensor:
        """(n_envs, 13): [q(4), dq(4), q_target(4), phase(1)]."""
        q_target = self._get_target()
        phase = (self.step_count.float() / self.max_steps).unsqueeze(-1)
        return torch.cat([self.q, self.dq, q_target, phase], dim=-1)

    def reset(self, seed: int | None = None) -> Tensor:
        """Reset all environments. Returns obs (n_envs, 13)."""
        if seed is not None:
            torch.manual_seed(seed)
        self.q.uniform_(-0.5, 0.5)
        self.dq.zero_()
        self.step_count.zero_()
        self._generate_trajectory_params()
        self.current_obs = self._build_obs()
        return self.current_obs

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Step all environments.

        Args:
            action: (n_envs, 4) raw policy output in [-1, 1], scaled internally.

        Returns:
            obs, reward, done, info.
        """
        tau = action.clamp(-1.0, 1.0) * self.action_scale

        # Semi-implicit Euler
        M = self._compute_mass_matrix_batch(self.q)
        rhs = tau - self.damping * self.dq
        qddot = torch.linalg.solve(M, rhs)
        self.dq = self.dq + self.dt * qddot
        self.q = self.q + self.dt * self.dq
        self.dq.clamp_(-5.0, 5.0)
        self.q.clamp_(-torch.pi, torch.pi)

        self.step_count += 1

        # Reward
        q_target = self._get_target()
        q_err = self.q - q_target
        reward = -(q_err**2).sum(dim=-1) - 0.01 * (tau**2).sum(dim=-1)
        rmse = q_err.pow(2).mean(dim=-1).sqrt()

        done = self.step_count >= self.max_steps

        # Auto-reset done envs
        if done.any():
            n = int(done.sum().item())
            self.q[done] = torch.randn(n, self.n_dof, device=self.device) * 0.3
            self.dq[done] = 0.0
            self.step_count[done] = 0
            self._generate_trajectory_params(done)

        self.current_obs = self._build_obs()
        return self.current_obs, reward, done, {"rmse": rmse}

    # ------------------------------------------------------------------
    # Coupling analysis
    # ------------------------------------------------------------------

    def get_coupling_info(self) -> dict[str, Tensor]:
        """Compute cross-arm coupling from current state.

        Returns:
            J_cross: (batch,2,2) normalized cross-arm coupling block
            sigma: (batch,2)     singular values
            U: (batch,2,2)       left singular vectors
            Vh: (batch,2,2)      right singular vectors
            J_cross_flat: (batch,4) flattened for obs augmentation
        """
        M = self._compute_mass_matrix_batch(self.q)
        diag = torch.diagonal(M, dim1=-2, dim2=-1)  # (batch, 4)
        scale = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2) + 1e-12)
        J_full = M / scale
        J_cross = J_full[:, :2, 2:]

        U, sigma, Vh = torch.linalg.svd(J_cross)
        return {
            "J_cross": J_cross,
            "sigma": sigma,
            "U": U,
            "Vh": Vh,
            "J_cross_flat": J_cross.reshape(-1, 4),
        }
