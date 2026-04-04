"""DualArmReach: 14-DOF bimanual tasks with coupling reward.

Three task modes:
- independent: Each arm reaches its own target (no coordination needed)
- transport_weld: Object welded to both EEs, move to target (Scheme A)
- transport_virtual: Virtual object = midpoint of EEs, maintain grasp width (Scheme B)

Observation (40-dim for all modes):
  independent:      [q_L(7), q_R(7), dq_L(7), dq_R(7), ee_L(3), ee_R(3), target_L(3), target_R(3)]
  transport_weld:   [q_L(7), q_R(7), dq_L(7), dq_R(7), ee_L(3), ee_R(3), obj_pos(3), target(3)]
  transport_virtual:[q_L(7), q_R(7), dq_L(7), dq_R(7), ee_L(3), ee_R(3), obj_pos(3), target(3)]

Action: tau(14), Box(-50, 50)
Episode: 200 steps (dt=0.002 -> 0.4s real time)

Coupling reward modes:
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
    """14-DOF dual-arm task with optional coupling reward shaping."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        task_mode: str = "independent",
        object_mass: float = 1.0,
        coupling_lambda: float = 0.0,
        grasp_width: float = 0.4,
        grasp_penalty: float = 5.0,
        target_radius_range: tuple[float, float] = (0.1, 0.3),
        max_episode_steps: int = 200,
        success_threshold: float = 0.10,
        reward_mode: str = "vanilla",
        quantum_computer: Any | None = None,
    ):
        super().__init__()
        assert task_mode in ("independent", "transport_weld", "transport_virtual")

        self.render_mode = render_mode
        self.task_mode = task_mode
        self.object_mass = object_mass
        self.coupling_lambda = coupling_lambda
        self.grasp_width = grasp_width
        self.grasp_penalty = grasp_penalty
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

        # Action and observation spaces (40-dim obs for all modes)
        self.action_space = spaces.Box(
            low=-50.0, high=50.0, shape=(N_JOINTS_TOTAL,), dtype=np.float32
        )
        obs_high = np.inf * np.ones(40, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Internal state
        self._target = np.zeros(3, dtype=float)     # transport modes: single target
        self._target_L = np.zeros(3, dtype=float)    # independent mode
        self._target_R = np.zeros(3, dtype=float)    # independent mode
        self._step_count = 0

        # Coupling cache (set by training loop per episode)
        self._cached_J = None
        self._cached_C = None
        self._cached_groups = None

        # Site / body IDs
        self._ee_L_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "L_ee"
        )
        self._ee_R_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "R_ee"
        )
        self._object_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )

        # Weld constraint IDs (for transport_weld mode)
        self._grasp_L_eq = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_left"
        )
        self._grasp_R_eq = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right"
        )

        # DOF indices: L_j1-L_j7=0-6, L_grip=7, R_j1-R_j7=8-14, R_grip=15
        # object_free: qpos[16:23] (3 pos + 4 quat), qvel[16:22] (6 dof)
        self._L_qpos = list(range(0, 7))
        self._R_qpos = list(range(8, 15))
        self._L_qvel = list(range(0, 7))
        self._R_qvel = list(range(8, 15))
        self._obj_qpos = slice(16, 23)  # 3 pos + 4 quat

        # Actuator indices
        self._L_ctrl = list(range(0, 7))
        self._R_ctrl = list(range(8, 15))

    # ── Observation ──────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        q_L = self.data.qpos[self._L_qpos].copy()
        q_R = self.data.qpos[self._R_qpos].copy()
        dq_L = self.data.qvel[self._L_qvel].copy()
        dq_R = self.data.qvel[self._R_qvel].copy()
        ee_L = self.data.site_xpos[self._ee_L_site].copy()
        ee_R = self.data.site_xpos[self._ee_R_site].copy()

        if self.task_mode == "independent":
            tail = np.concatenate([self._target_L, self._target_R])
        else:
            obj_pos = self._get_object_pos()
            tail = np.concatenate([obj_pos, self._target])

        return np.concatenate([
            q_L, q_R, dq_L, dq_R, ee_L, ee_R, tail,
        ]).astype(np.float32)

    def _get_object_pos(self) -> np.ndarray:
        """Object position: MuJoCo body (weld) or EE midpoint (virtual)."""
        if self.task_mode == "transport_weld":
            return self.data.xpos[self._object_body].copy()
        else:  # transport_virtual
            ee_L = self.data.site_xpos[self._ee_L_site]
            ee_R = self.data.site_xpos[self._ee_R_site]
            return ((ee_L + ee_R) / 2.0).copy()

    # ── Target sampling ──────────────────────────────────────────────

    def _sample_transport_target(self) -> np.ndarray:
        """Sample reachable target for object transport (between the arms)."""
        rng = self.np_random
        r_min, r_max = self.target_radius_range
        r = rng.uniform(r_min, r_max)
        # Target in front of the torso, between the two arm bases
        x = rng.uniform(-0.15, 0.15)
        y = rng.uniform(0.15, 0.45)
        z = rng.uniform(0.4, 0.9)
        return np.array([x, y, z], dtype=float)

    def _sample_arm_target(self, side: str) -> np.ndarray:
        """Sample target for one arm (independent mode)."""
        rng = self.np_random
        r_min, r_max = self.target_radius_range
        r = rng.uniform(r_min, r_max)
        theta = rng.uniform(0.3, np.pi - 0.3)
        phi = rng.uniform(0, 2 * np.pi)
        if side == "left":
            base_x, base_y, base_z = -0.2, 0.0, 0.796
        else:
            base_x, base_y, base_z = 0.2, 0.0, 0.796
        x = base_x + r * np.sin(theta) * np.cos(phi)
        y = base_y + r * np.sin(theta) * np.sin(phi)
        z = max(base_z + r * np.cos(theta), 0.1)
        return np.array([x, y, z], dtype=float)

    # ── Coupling reward ──────────────────────────────────────────────

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
        J = self._cached_J
        if J is None:
            return 0.0
        a = action / 50.0
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

    # ── Reset / Step ─────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._cached_J = None
        self._cached_C = None
        self._cached_groups = None

        mujoco.mj_resetData(self.model, self.data)

        # Arm joints: small random perturbation
        self.data.qpos[self._L_qpos] = self.np_random.uniform(-0.1, 0.1, size=7)
        self.data.qpos[self._R_qpos] = self.np_random.uniform(-0.1, 0.1, size=7)
        self.data.qvel[self._L_qvel] = 0.0
        self.data.qvel[self._R_qvel] = 0.0

        if self.task_mode == "transport_weld":
            # Activate weld constraints
            self.model.eq_active0[self._grasp_L_eq] = True
            self.model.eq_active0[self._grasp_R_eq] = True
            # Place object between the EEs
            mujoco.mj_forward(self.model, self.data)
            ee_L = self.data.site_xpos[self._ee_L_site]
            ee_R = self.data.site_xpos[self._ee_R_site]
            obj_init = (ee_L + ee_R) / 2.0
            self.data.qpos[16:19] = obj_init  # object position
            self.data.qpos[19:23] = [1, 0, 0, 0]  # identity quaternion
            self._target = self._sample_transport_target()
        elif self.task_mode == "transport_virtual":
            # Deactivate weld constraints (virtual object)
            self.model.eq_active0[self._grasp_L_eq] = False
            self.model.eq_active0[self._grasp_R_eq] = False
            self._target = self._sample_transport_target()
        else:  # independent
            self.model.eq_active0[self._grasp_L_eq] = False
            self.model.eq_active0[self._grasp_R_eq] = False
            self._target_L = self._sample_arm_target("left")
            self._target_R = self._sample_arm_target("right")

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        obs = self._get_obs()
        info = {"target": self._target.copy()} if self.task_mode != "independent" else {
            "target_L": self._target_L.copy(),
            "target_R": self._target_R.copy(),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -50.0, 50.0).astype(float)
        self.data.ctrl[self._L_ctrl] = action[:7]
        self.data.ctrl[self._R_ctrl] = action[7:]
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        ee_L = self.data.site_xpos[self._ee_L_site]
        ee_R = self.data.site_xpos[self._ee_R_site]

        if self.task_mode == "independent":
            reward, success, info = self._compute_independent_reward(action, ee_L, ee_R)
        elif self.task_mode == "transport_weld":
            reward, success, info = self._compute_weld_reward(action, ee_L, ee_R)
        else:  # transport_virtual
            reward, success, info = self._compute_virtual_reward(action, ee_L, ee_R)

        reward += self._compute_coupling_reward(action)

        terminated = success
        truncated = self._step_count >= self.max_episode_steps
        return obs, float(reward), terminated, truncated, info

    def _compute_independent_reward(self, action, ee_L, ee_R):
        dist_L = np.linalg.norm(ee_L - self._target_L)
        dist_R = np.linalg.norm(ee_R - self._target_R)
        reward = -(dist_L + dist_R) - 0.01 * np.sum(action**2)
        success = dist_L < self.success_threshold and dist_R < self.success_threshold
        info = {"dist_L": dist_L, "dist_R": dist_R, "success": success}
        return reward, success, info

    def _compute_weld_reward(self, action, ee_L, ee_R):
        obj_pos = self.data.xpos[self._object_body]
        dist = np.linalg.norm(obj_pos - self._target)
        reward = -dist - 0.01 * np.sum(action**2)
        success = dist < self.success_threshold
        info = {"dist": dist, "success": success, "obj_pos": obj_pos.copy()}
        return reward, success, info

    def _compute_virtual_reward(self, action, ee_L, ee_R):
        obj_pos = (ee_L + ee_R) / 2.0
        dist = np.linalg.norm(obj_pos - self._target)
        # Grasp width maintenance: penalize deviation from desired EE separation
        ee_sep = ee_L - ee_R
        desired_sep = np.array([-self.grasp_width, 0, 0])  # L is at -x, R at +x
        grasp_err = np.linalg.norm(ee_sep - desired_sep)
        reward = -dist - self.grasp_penalty * grasp_err - 0.01 * np.sum(action**2)
        success = dist < self.success_threshold and grasp_err < 0.05
        info = {
            "dist": dist, "grasp_err": grasp_err,
            "success": success, "obj_pos": obj_pos.copy(),
        }
        return reward, success, info

    def render(self) -> np.ndarray | None:
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
