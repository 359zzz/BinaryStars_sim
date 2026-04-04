"""DualArmReach-v0: 14-DOF bimanual reaching with coupling reward.

Two OpenArm v10 7-DOF arms mounted on a shared torso.
Each arm reaches its own target; coupling reward shapes cross-arm coordination.

Observation: [q_L(7), q_R(7), dq_L(7), dq_R(7), ee_L(3), ee_R(3),
              target_L(3), target_R(3)] = 40-dim
Action: tau(14), Box(-50, 50)
Reward: -(||ee_L - tgt_L|| + ||ee_R - tgt_R||) - 0.01*||tau||^2
        [+ optional coupling reward]
Success: ||ee_L - tgt_L|| < thresh AND ||ee_R - tgt_R|| < thresh
Episode: 200 steps (dt=0.002 -> 0.4s real time)

Reward modes:
- vanilla: no coupling reward
- classical_coupling: weighted by |J_ij| (from M_eff)
- quantum_entanglement: weighted by C_ij (n-body quantum)
- quantum_decomposed: entanglement clustering + within-group variance
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

N_JOINTS_PER_ARM = 7
N_JOINTS_TOTAL = 14

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "openarm14_dual.xml")


class DualArmReachEnv(gym.Env):
    """14-DOF dual-arm reaching task with optional coupling reward shaping."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        object_mass: float = 1.0,
        coupling_lambda: float = 0.0,
        target_radius_range: tuple[float, float] = (0.2, 0.5),
        max_episode_steps: int = 200,
        success_threshold: float = 0.10,
        reward_mode: str = "vanilla",
        quantum_computer: Any | None = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.object_mass = object_mass
        self.coupling_lambda = coupling_lambda
        self.target_radius_range = target_radius_range
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.reward_mode = reward_mode
        self._quantum_computer = quantum_computer

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(
            os.path.normpath(_MODEL_PATH)
        )
        self.data = mujoco.MjData(self.model)

        # Renderer
        self.renderer = None
        if render_mode == "human":
            self.renderer = mujoco.Renderer(self.model)

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-50.0, high=50.0, shape=(N_JOINTS_TOTAL,), dtype=np.float32
        )
        # obs = [q_L(7), q_R(7), dq_L(7), dq_R(7), ee_L(3), ee_R(3),
        #        target_L(3), target_R(3)] = 40-dim
        obs_high = np.inf * np.ones(40, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Internal state
        self._target_L = np.zeros(3, dtype=float)
        self._target_R = np.zeros(3, dtype=float)
        self._step_count = 0

        # Coupling cache (set by training loop per episode)
        self._cached_J = None
        self._cached_C = None
        self._cached_groups = None

        # Site IDs
        self._ee_L_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "L_ee"
        )
        self._ee_R_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "R_ee"
        )

        # DOF indices: L_j1-L_j7=0-6, L_grip=7, R_j1-R_j7=8-14, R_grip=15
        self._L_qpos = list(range(0, 7))
        self._R_qpos = list(range(8, 15))
        self._L_qvel = list(range(0, 7))
        self._R_qvel = list(range(8, 15))

        # Actuator indices: a_L_j1-a_L_j7=0-6, a_L_grip=7,
        #                    a_R_j1-a_R_j7=8-14, a_R_grip=15
        self._L_ctrl = list(range(0, 7))
        self._R_ctrl = list(range(8, 15))

    def _get_obs(self) -> np.ndarray:
        q_L = self.data.qpos[self._L_qpos].copy()
        q_R = self.data.qpos[self._R_qpos].copy()
        dq_L = self.data.qvel[self._L_qvel].copy()
        dq_R = self.data.qvel[self._R_qvel].copy()
        ee_L = self.data.site_xpos[self._ee_L_site].copy()
        ee_R = self.data.site_xpos[self._ee_R_site].copy()
        return np.concatenate([
            q_L, q_R, dq_L, dq_R, ee_L, ee_R,
            self._target_L, self._target_R,
        ]).astype(np.float32)

    def _sample_target(self, side: str) -> np.ndarray:
        """Sample target in reachable workspace for one arm."""
        rng = self.np_random
        r_min, r_max = self.target_radius_range
        r = rng.uniform(r_min, r_max)
        theta = rng.uniform(0.3, np.pi - 0.3)  # avoid poles
        phi = rng.uniform(0, 2 * np.pi)

        # Arm base positions (from MJCF)
        if side == "left":
            base_x, base_y, base_z = -0.2, 0.0, 0.098 + 0.698
        else:
            base_x, base_y, base_z = 0.2, 0.0, 0.098 + 0.698

        x = base_x + r * np.sin(theta) * np.cos(phi)
        y = base_y + r * np.sin(theta) * np.sin(phi)
        z = base_z + r * np.cos(theta)
        z = max(z, 0.1)  # above ground
        return np.array([x, y, z], dtype=float)

    def set_cached_coupling(
        self,
        J: np.ndarray | None = None,
        C: np.ndarray | None = None,
        groups: list | None = None,
    ) -> None:
        """Set episode-level cached coupling matrices."""
        self._cached_J = J
        self._cached_C = C
        self._cached_groups = groups

    def _compute_coupling_reward(self, action: np.ndarray) -> float:
        """Dispatch to reward mode."""
        if self.coupling_lambda <= 0 or self.reward_mode == "vanilla":
            return 0.0

        if self.reward_mode == "classical_coupling":
            return self._coupling_reward_classical(action)
        elif self.reward_mode == "quantum_entanglement":
            return self._coupling_reward_quantum(action)
        elif self.reward_mode == "quantum_decomposed":
            return self._coupling_reward_quantum_decomposed(action)
        return 0.0

    def _coupling_reward_classical(self, action: np.ndarray) -> float:
        """Classical coupling reward from M_eff: weighted by |J_ij|."""
        J = self._cached_J
        if J is None:
            return 0.0  # No cache → skip (will be set at next reset)
        a = action / 50.0  # normalize to [-1, 1]

        penalty = 0.0
        Z = 0.0
        for i in range(N_JOINTS_TOTAL):
            for j in range(i + 1, N_JOINTS_TOTAL):
                w = abs(J[i, j])
                Z += w
                sign_j = np.sign(J[i, j]) if abs(J[i, j]) > 1e-6 else 0.0
                penalty += w * (a[i] - sign_j * a[j]) ** 2

        if Z > 1e-8:
            penalty /= Z
        return -self.coupling_lambda * penalty

    def _coupling_reward_quantum(self, action: np.ndarray) -> float:
        """Quantum entanglement reward: weighted by C_ij (n-body)."""
        C = self._cached_C
        J = self._cached_J
        if C is None:
            return 0.0
        if J is None:
            J = np.zeros_like(C)
        a = action / 50.0

        penalty = 0.0
        Z = 0.0
        for i in range(N_JOINTS_TOTAL):
            for j in range(i + 1, N_JOINTS_TOTAL):
                w = C[i, j]
                if w < 1e-8:
                    continue
                Z += w
                if abs(J[i, j]) > 1e-6:
                    sign_j = np.sign(J[i, j])
                else:
                    sign_j = 1.0  # indirect coupling: cooperative
                penalty += w * (a[i] - sign_j * a[j]) ** 2

        if Z > 1e-8:
            penalty /= Z
        return -self.coupling_lambda * penalty

    def _coupling_reward_quantum_decomposed(self, action: np.ndarray) -> float:
        """Quantum decomposed reward: entanglement clustering + within-group variance."""
        groups = self._cached_groups
        if groups is None:
            return 0.0
        a = action / 50.0

        penalty = 0.0
        n_groups = 0
        for group in groups:
            if len(group) < 2:
                continue
            penalty += np.var(a[group])
            n_groups += 1

        if n_groups > 0:
            penalty /= n_groups
        return -self.coupling_lambda * penalty

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Clear episode-level coupling cache
        self._cached_J = None
        self._cached_C = None
        self._cached_groups = None

        mujoco.mj_resetData(self.model, self.data)

        # Random initial joint positions (small perturbation around home)
        self.data.qpos[self._L_qpos] = self.np_random.uniform(-0.1, 0.1, size=7)
        self.data.qpos[self._R_qpos] = self.np_random.uniform(-0.1, 0.1, size=7)
        self.data.qvel[self._L_qvel] = 0.0
        self.data.qvel[self._R_qvel] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Sample targets for both arms
        self._target_L = self._sample_target("left")
        self._target_R = self._sample_target("right")

        self._step_count = 0
        obs = self._get_obs()
        info = {
            "target_L": self._target_L.copy(),
            "target_R": self._target_R.copy(),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -50.0, 50.0).astype(float)
        # Map 14-dim action to 16-dim ctrl (skip grip actuators at indices 7, 15)
        self.data.ctrl[self._L_ctrl] = action[:7]
        self.data.ctrl[self._R_ctrl] = action[7:]
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        ee_L = self.data.site_xpos[self._ee_L_site]
        ee_R = self.data.site_xpos[self._ee_R_site]
        dist_L = np.linalg.norm(ee_L - self._target_L)
        dist_R = np.linalg.norm(ee_R - self._target_R)

        reward = -(dist_L + dist_R) - 0.01 * np.sum(action**2)
        reward += self._compute_coupling_reward(action)

        success_L = dist_L < self.success_threshold
        success_R = dist_R < self.success_threshold
        success = success_L and success_R
        terminated = success
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "dist_L": dist_L,
            "dist_R": dist_R,
            "success": success,
            "success_L": success_L,
            "success_R": success_R,
            "ee_L": ee_L.copy(),
            "ee_R": ee_R.copy(),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
