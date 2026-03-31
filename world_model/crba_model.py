"""CRBA physics model — zero-shot transfer via MuJoCo's internal CRBA.

Uses mj_fullM + mj_bias for forward prediction.
Key: only body_mass/inertia parameters change with payload, no retraining.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "openarm7.xml"
)
N_JOINTS = 7


class CRBAModel:
    """Physics-based world model using MuJoCo's internal dynamics."""

    def __init__(self, payload_kg: float = 0.0):
        self.model = mujoco.MjModel.from_xml_path(os.path.normpath(_MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.payload_kg = payload_kg

        if payload_kg > 0:
            self._apply_payload(payload_kg)

    def _apply_payload(self, payload_kg: float) -> None:
        from physics.openarm_params import modify_payload
        masses_new, inertias_new, coms_new = modify_payload(payload_kg)
        link6_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link6"
        )
        self.model.body_mass[link6_id] = masses_new[6]
        self.model.body_ipos[link6_id] = coms_new[6]
        self.model.body_inertia[link6_id] = np.diag(inertias_new[6])

    def predict_batch(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Predict next state for a batch of transitions.

        Uses MuJoCo's mj_fullM and mj_bias for:
            ddq = M(q)^{-1} * (tau - bias(q, dq))
            dq_next = dq + ddq * dt
            q_next = q + dq_next * dt  (semi-implicit Euler)
        """
        q = data["q"]
        dq = data["dq"]
        tau = data["tau"]
        n = q.shape[0]
        dt = self.model.opt.timestep

        q_pred = np.zeros_like(q)
        dq_pred = np.zeros_like(dq)

        nv = self.model.nv
        M_full = np.zeros((nv, nv))

        for i in range(n):
            self.data.qpos[:N_JOINTS] = q[i]
            self.data.qvel[:N_JOINTS] = dq[i]
            mujoco.mj_forward(self.model, self.data)

            # Full mass matrix
            mujoco.mj_fullM(self.model, M_full, self.data.qM)
            M = M_full[:N_JOINTS, :N_JOINTS]

            # Bias forces: C(q,dq)*dq + g(q)
            bias = self.data.qfrc_bias[:N_JOINTS].copy()

            # Forward dynamics
            rhs = tau[i] - bias
            ddq = np.linalg.solve(M, rhs)

            # Semi-implicit Euler
            dq_pred[i] = dq[i] + ddq * dt
            q_pred[i] = q[i] + dq_pred[i] * dt

        return {"q_next": q_pred, "dq_next": dq_pred}

    def set_payload(self, payload_kg: float) -> None:
        """Change payload — instant transfer, no retraining."""
        self.__init__(payload_kg=payload_kg)
