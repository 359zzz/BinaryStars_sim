"""Collect transition data from MuJoCo simulation.

Transition format: (q, dq, tau) -> (q_next, dq_next)
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "openarm7.xml"
)
N_JOINTS = 7


def collect_transitions(
    n_rollouts: int = 200,
    rollout_len: int = 100,
    payload_kg: float = 0.0,
    seed: int = 0,
    torque_scale: float = 20.0,
) -> dict[str, np.ndarray]:
    """Collect transition data with random torque actions.

    Returns dict with keys: q, dq, tau, q_next, dq_next
    Each array has shape (n_rollouts * rollout_len, 7).
    """
    rng = np.random.RandomState(seed)

    model = mujoco.MjModel.from_xml_path(os.path.normpath(_MODEL_PATH))
    data = mujoco.MjData(model)

    # Apply payload to link6
    if payload_kg > 0:
        from physics.openarm_params import modify_payload
        masses_new, inertias_new, coms_new = modify_payload(payload_kg)
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        model.body_mass[link6_id] = masses_new[6]
        model.body_ipos[link6_id] = coms_new[6]
        model.body_inertia[link6_id] = np.diag(inertias_new[6])

    total = n_rollouts * rollout_len
    q_all = np.zeros((total, N_JOINTS), dtype=np.float32)
    dq_all = np.zeros((total, N_JOINTS), dtype=np.float32)
    tau_all = np.zeros((total, N_JOINTS), dtype=np.float32)
    q_next_all = np.zeros((total, N_JOINTS), dtype=np.float32)
    dq_next_all = np.zeros((total, N_JOINTS), dtype=np.float32)

    idx = 0
    for ep in range(n_rollouts):
        mujoco.mj_resetData(model, data)
        # Random initial joint positions within limits
        for j in range(N_JOINTS):
            lo = model.jnt_range[j, 0]
            hi = model.jnt_range[j, 1]
            data.qpos[j] = rng.uniform(lo * 0.5, hi * 0.5)
        data.qvel[:N_JOINTS] = rng.uniform(-0.5, 0.5, size=N_JOINTS)
        mujoco.mj_forward(model, data)

        for t in range(rollout_len):
            q_all[idx] = data.qpos[:N_JOINTS]
            dq_all[idx] = data.qvel[:N_JOINTS]

            # Random torque (smooth: hold for a few steps)
            if t % 10 == 0:
                tau = rng.uniform(-torque_scale, torque_scale, size=N_JOINTS)
            tau_all[idx] = tau
            data.ctrl[:N_JOINTS] = tau

            mujoco.mj_step(model, data)

            q_next_all[idx] = data.qpos[:N_JOINTS]
            dq_next_all[idx] = data.qvel[:N_JOINTS]
            idx += 1

    return {
        "q": q_all,
        "dq": dq_all,
        "tau": tau_all,
        "q_next": q_next_all,
        "dq_next": dq_next_all,
    }


def collect_all_payloads(
    payloads: list[float] | None = None,
    n_rollouts: int = 200,
    rollout_len: int = 100,
    seed: int = 0,
    save_dir: str | None = None,
) -> dict[float, dict[str, np.ndarray]]:
    """Collect data for multiple payload conditions."""
    if payloads is None:
        payloads = [0.0, 0.5, 1.0]

    all_data = {}
    for i, p in enumerate(payloads):
        print(f"Collecting payload={p:.1f} kg ({n_rollouts} rollouts)...")
        data = collect_transitions(
            n_rollouts=n_rollouts,
            rollout_len=rollout_len,
            payload_kg=p,
            seed=seed + i * 1000,
        )
        all_data[p] = data

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"transitions_payload_{p:.1f}.npz")
            np.savez_compressed(path, **data)
            print(f"  Saved {path} ({data['q'].shape[0]} transitions)")

    return all_data
