"""OpenArmReach-v0: 7-DOF reaching with payload switching and coupling reward.

Observation: [q(7), dq(7), ee_pos(3), target_pos(3)] = 20-dim
Action: tau(7), Box(-50, 50)
Reward: -||ee - target|| - 0.01*||tau||^2 [+ optional coupling reward]
Success: ||ee - target|| < 0.05 m
Episode: 200 steps (dt=0.002 -> 0.4s real time)

Reward modes:
- vanilla: no coupling reward
- classical_coupling: weighted by |J_ij| (pairwise classical)
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

from physics.openarm_params import (
    N_JOINTS,
    compute_openarm_coupling,
    compute_openarm_mass_matrix,
    modify_payload,
)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "openarm7.xml")


class OpenArmReachEnv(gym.Env):
    """7-DOF OpenArm reaching task with optional coupling reward shaping."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        payload_kg: float = 0.0,
        coupling_lambda: float = 0.0,
        target_radius_range: tuple[float, float] = (0.3, 0.6),
        max_episode_steps: int = 200,
        success_threshold: float = 0.05,
        reward_mode: str = "vanilla",
        quantum_computer: Any | None = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.payload_kg = payload_kg
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

        # Apply payload
        if payload_kg > 0:
            self._apply_payload(payload_kg)

        # Renderer
        self.renderer = None
        if render_mode == "human":
            self.renderer = mujoco.Renderer(self.model)

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-50.0, high=50.0, shape=(N_JOINTS,), dtype=np.float32
        )
        # obs = [q(7), dq(7), ee_pos(3), target_pos(3)]
        obs_high = np.inf * np.ones(20, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Internal state
        self._target_pos = np.zeros(3, dtype=float)
        self._step_count = 0

        # Site / body IDs
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee"
        )
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

    def _apply_payload(self, payload_kg: float) -> None:
        """Modify MuJoCo model to add payload at EE."""
        link6_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link6"
        )
        masses_new, inertias_new, coms_new = modify_payload(payload_kg)

        self.model.body_mass[link6_id] = masses_new[6]
        self.model.body_ipos[link6_id] = coms_new[6]
        self.model.body_inertia[link6_id] = np.diag(inertias_new[6])

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[:N_JOINTS].copy()
        dq = self.data.qvel[:N_JOINTS].copy()
        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        return np.concatenate([q, dq, ee_pos, self._target_pos]).astype(np.float32)

    def _sample_target(self) -> np.ndarray:
        """Sample target on sphere in reachable workspace."""
        rng = self.np_random
        r_min, r_max = self.target_radius_range
        r = rng.uniform(r_min, r_max)
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        base_z = 0.698
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = base_z + r * np.cos(theta)
        z = max(z, 0.1)
        return np.array([x, y, z], dtype=float)

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
        """Classical coupling reward: weighted by |J_ij|."""
        q = self.data.qpos[:N_JOINTS]
        J = compute_openarm_coupling(q)

        penalty = 0.0
        Z = 0.0
        for i in range(N_JOINTS):
            for j in range(i + 1, N_JOINTS):
                w = abs(J[i, j])
                Z += w
                sign_j = np.sign(J[i, j]) if abs(J[i, j]) > 1e-6 else 0.0
                penalty += w * (action[i] - sign_j * action[j]) ** 2

        if Z > 1e-8:
            penalty /= Z
        return -self.coupling_lambda * penalty

    def _coupling_reward_quantum(self, action: np.ndarray) -> float:
        """Quantum entanglement reward: weighted by C_ij (n-body).

        Key difference from classical: indirect coupling pairs (J_ij ~ 0 but C_ij > 0)
        still contribute to coordination penalty.
        """
        q = self.data.qpos[:N_JOINTS]
        if self._quantum_computer is None:
            return 0.0

        C = self._quantum_computer.get_entanglement_graph(q)
        J = self._quantum_computer.get_classical_coupling(q)

        penalty = 0.0
        Z = 0.0
        for i in range(N_JOINTS):
            for j in range(i + 1, N_JOINTS):
                w = C[i, j]  # quantum weight (includes indirect coupling)
                Z += w
                sign_j = np.sign(J[i, j]) if abs(J[i, j]) > 1e-6 else 0.0
                penalty += w * (action[i] - sign_j * action[j]) ** 2

        if Z > 1e-8:
            penalty /= Z
        return -self.coupling_lambda * penalty

    def _coupling_reward_quantum_decomposed(self, action: np.ndarray) -> float:
        """Quantum decomposed reward: entanglement clustering + within-group variance."""
        q = self.data.qpos[:N_JOINTS]
        if self._quantum_computer is None:
            return 0.0

        C = self._quantum_computer.get_entanglement_graph(q)

        from quantum_prior.clustering import decompose_joints
        groups = decompose_joints(C)

        penalty = 0.0
        n_groups = 0
        for group in groups:
            if len(group) < 2:
                continue
            group_actions = action[group]
            penalty += np.var(group_actions)
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

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:N_JOINTS] = self.np_random.uniform(
            -0.1, 0.1, size=N_JOINTS
        )
        self.data.qvel[:N_JOINTS] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._target_pos = self._sample_target()
        if self.model.nmocap > 0:
            self.data.mocap_pos[0] = self._target_pos

        self._step_count = 0
        obs = self._get_obs()
        info = {"target_pos": self._target_pos.copy()}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -50.0, 50.0).astype(float)
        self.data.ctrl[:N_JOINTS] = action
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        ee_pos = self.data.site_xpos[self._ee_site_id]
        dist = np.linalg.norm(ee_pos - self._target_pos)

        reward = -dist - 0.01 * np.sum(action**2)
        reward += self._compute_coupling_reward(action)

        success = dist < self.success_threshold
        terminated = success
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "distance": dist,
            "success": success,
            "ee_pos": ee_pos.copy(),
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

    def set_payload(self, payload_kg: float) -> None:
        """Runtime payload change (for transfer experiments)."""
        self.payload_kg = payload_kg
        self.model = mujoco.MjModel.from_xml_path(
            os.path.normpath(_MODEL_PATH)
        )
        self.data = mujoco.MjData(self.model)
        if payload_kg > 0:
            self._apply_payload(payload_kg)
