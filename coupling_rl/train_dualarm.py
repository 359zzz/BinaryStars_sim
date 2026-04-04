"""Training script for 14-DOF dual-arm coupling-aware RL.

Four PPO variants:
1. vanilla:        Standard MLP policy, no coupling reward
2. coupling:       Classical |J_ij| from M_eff, 14-DOF coupling reward
3. quantum_c:      Quantum C_ij from M_eff entanglement, n-body reward
4. quantum_decomp: Quantum entanglement clustering + within-group variance

Usage:
    python -m coupling_rl.train_dualarm --config configs/dualarm.yaml --variant vanilla --seed 0
"""

from __future__ import annotations

# MUST be before numpy/torch/scipy imports — prevents BLAS thread deadlock
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import time

import numpy as np
import torch
import yaml

from coupling_rl.networks import (
    CouplingAwarePolicy,
    QuantumDecomposedPolicy,
    ValueNet,
    VanillaPolicy,
)
from coupling_rl.ppo import PPOConfig, RolloutBuffer, ppo_update
from envs.dualarm_reach import DualArmReachEnv, N_JOINTS_TOTAL

ALL_VARIANTS = ["vanilla", "coupling", "quantum_c", "quantum_decomp"]

# 14 joints → 14*13/2 = 91 unique pairs
N_COUPLING_FEATURES = N_JOINTS_TOTAL * (N_JOINTS_TOTAL - 1) // 2
BASE_OBS_DIM = 40
AUG_OBS_DIM = BASE_OBS_DIM + N_COUPLING_FEATURES  # 40 + 91 = 131

# Variant -> (reward_mode, needs_quantum, obs_dim)
VARIANT_CONFIG = {
    "vanilla":        ("vanilla",              False, BASE_OBS_DIM),
    "coupling":       ("classical_coupling",   True,  AUG_OBS_DIM),
    "quantum_c":      ("quantum_entanglement", True,  AUG_OBS_DIM),
    "quantum_decomp": ("quantum_decomposed",   True,  AUG_OBS_DIM),
}


def _make_quantum_computer(cfg: dict):
    """Create CachedEntanglementComputer for 14-DOF dual-arm M_eff."""
    from physics.dualarm_mass import make_dualarm_mass_fn
    from quantum_prior.cached_computer import CachedEntanglementComputer

    mass_fn = make_dualarm_mass_fn(
        object_mass=cfg.get("object_mass", 1.0),
    )
    return CachedEntanglementComputer(
        mass_matrix_fn=mass_fn,
        resolution=cfg.get("quantum_resolution", 0.3),
        cache_size=cfg.get("quantum_cache_size", 200000),
    )


def _compute_episode_features(
    q: np.ndarray,
    variant: str,
    quantum_computer,
    env=None,
) -> np.ndarray | None:
    """Compute structure features once per episode reset.

    Returns flat upper-triangle features (91-dim for 14 joints).
    Also sets cached coupling on env for reward computation.
    """
    if variant == "vanilla" or quantum_computer is None:
        return None
    if variant == "coupling":
        J = quantum_computer.get_classical_coupling(q)
        if env is not None:
            env.set_cached_coupling(J=J)
        return quantum_computer.get_classical_features(q)
    else:  # quantum_c, quantum_decomp
        C = quantum_computer.get_entanglement_graph(q)
        J = quantum_computer.get_classical_coupling(q)
        groups = None
        if variant == "quantum_decomp":
            from quantum_prior.clustering import decompose_joints
            groups = decompose_joints(C)
        if env is not None:
            env.set_cached_coupling(J=J, C=C, groups=groups)
        return quantum_computer.get_entanglement_features(q)


def _augment_obs(
    obs_base: np.ndarray,
    env_features: list[np.ndarray | None],
    variant: str,
) -> np.ndarray:
    """Augment obs with precomputed episode-level features."""
    if variant == "vanilla":
        return obs_base
    n_envs = obs_base.shape[0]
    feats = np.zeros((n_envs, N_COUPLING_FEATURES), dtype=np.float32)
    for i in range(n_envs):
        if env_features[i] is not None:
            feats[i] = env_features[i]
    return np.concatenate([obs_base, feats], axis=1)


def make_envs(
    n_envs: int,
    variant: str,
    coupling_lambda: float,
    object_mass: float,
    seed: int,
    quantum_computer=None,
    task_mode: str = "independent",
    grasp_width: float = 0.4,
    grasp_penalty: float = 5.0,
) -> list[DualArmReachEnv]:
    """Create dual-arm environments."""
    reward_mode, needs_quantum, _ = VARIANT_CONFIG[variant]
    lam = coupling_lambda if reward_mode != "vanilla" else 0.0

    envs = []
    for i in range(n_envs):
        env = DualArmReachEnv(
            task_mode=task_mode,
            object_mass=object_mass,
            coupling_lambda=lam,
            grasp_width=grasp_width,
            grasp_penalty=grasp_penalty,
            reward_mode=reward_mode,
            quantum_computer=quantum_computer if needs_quantum else None,
        )
        env.reset(seed=seed + i)
        envs.append(env)
    return envs


def collect_rollout(
    envs: list[DualArmReachEnv],
    policy: torch.nn.Module,
    value_net: ValueNet,
    buffer: RolloutBuffer,
    config: PPOConfig,
    variant: str = "vanilla",
    quantum_computer=None,
) -> dict[str, float]:
    """Collect experience with episode-level quantum feature caching."""
    n_envs = len(envs)
    base_obs_dim = envs[0].observation_space.shape[0]

    obs_base = np.zeros((n_envs, base_obs_dim), dtype=np.float32)
    env_features: list[np.ndarray | None] = [None] * n_envs

    for i, env in enumerate(envs):
        if not hasattr(env, "_current_obs"):
            obs, _ = env.reset()
            env._current_obs = obs
        obs_base[i] = env._current_obs
        env_features[i] = _compute_episode_features(
            obs_base[i, :N_JOINTS_TOTAL], variant, quantum_computer, env=env,
        )

    ep_rewards = []
    ep_lengths = []
    ep_successes = []
    current_ep_reward = np.zeros(n_envs)
    current_ep_len = np.zeros(n_envs, dtype=int)

    buffer.reset()
    policy.eval()
    value_net.eval()

    with torch.no_grad():
        for step in range(config.n_steps):
            obs_np = _augment_obs(obs_base, env_features, variant)
            obs_t = torch.from_numpy(obs_np)

            action, log_prob = policy.get_action(obs_t)
            value = value_net(obs_t)

            action_np = action.numpy()
            log_prob_np = log_prob.numpy()
            value_np = value.numpy()

            action_scaled = np.clip(action_np * config.action_scale, -50.0, 50.0)

            next_obs_base = np.zeros_like(obs_base)
            rewards = np.zeros(n_envs)
            dones = np.zeros(n_envs)

            for i, env in enumerate(envs):
                obs_new, r, terminated, truncated, info = env.step(action_scaled[i])
                rewards[i] = r
                done = terminated or truncated
                dones[i] = float(done)

                current_ep_reward[i] += r
                current_ep_len[i] += 1

                if done:
                    ep_rewards.append(current_ep_reward[i])
                    ep_lengths.append(current_ep_len[i])
                    ep_successes.append(info.get("success", False))
                    current_ep_reward[i] = 0.0
                    current_ep_len[i] = 0
                    obs_new, _ = env.reset()
                    env_features[i] = _compute_episode_features(
                        obs_new[:N_JOINTS_TOTAL], variant, quantum_computer, env=env,
                    )

                next_obs_base[i] = obs_new
                env._current_obs = obs_new

            buffer.add(obs_np, action_np, rewards, dones, log_prob_np, value_np)
            obs_base = next_obs_base

        last_obs_np = _augment_obs(obs_base, env_features, variant)
        last_obs_t = torch.from_numpy(last_obs_np)
        last_value = value_net(last_obs_t).numpy()
        buffer.compute_gae(last_value, config.gamma, config.gae_lambda)

    metrics = {
        "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "success_rate": float(np.mean(ep_successes)) if ep_successes else 0.0,
        "n_episodes": len(ep_rewards),
    }
    return metrics


def train(
    variant: str = "vanilla",
    config_path: str | None = None,
    seed: int = 0,
) -> dict:
    """Train one PPO variant on dual-arm task."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "dualarm.yaml"
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    reward_mode, needs_quantum, obs_dim = VARIANT_CONFIG[variant]
    act_dim = N_JOINTS_TOTAL

    ppo_cfg = PPOConfig(
        clip_eps=cfg.get("clip_eps", 0.2),
        lr_policy=cfg.get("lr_policy", 3e-4),
        lr_value=cfg.get("lr_value", 1e-3),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        entropy_coef=cfg.get("entropy_coef", 0.01),
        n_epochs=cfg.get("ppo_epochs", 4),
        mini_batch_size=cfg.get("mini_batch_size", 2048),
        n_steps=cfg.get("n_steps", 2048),
        n_envs=cfg.get("n_envs", 8),
        total_timesteps=cfg.get("total_timesteps", 5_000_000),
    )
    coupling_lambda = cfg.get("coupling_lambda", 1.0)
    object_mass = cfg.get("object_mass", 1.0)
    task_mode = cfg.get("task_mode", "independent")
    grasp_width = cfg.get("grasp_width", 0.4)
    grasp_penalty = cfg.get("grasp_penalty", 5.0)

    device = "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    quantum_computer = _make_quantum_computer(cfg) if needs_quantum else None

    envs = make_envs(
        ppo_cfg.n_envs, variant, coupling_lambda, object_mass, seed,
        quantum_computer=quantum_computer,
        task_mode=task_mode,
        grasp_width=grasp_width,
        grasp_penalty=grasp_penalty,
    )

    # Networks — larger hidden dim for 14-DOF
    hidden_dim = cfg.get("hidden_dim", 256)
    if variant in ("quantum_c", "coupling"):
        policy = CouplingAwarePolicy(obs_dim, act_dim, hidden=hidden_dim).to(device)
    elif variant == "quantum_decomp":
        groups = None
        if quantum_computer is not None:
            q0 = np.zeros(N_JOINTS_TOTAL)
            C = quantum_computer.get_entanglement_graph(q0)
            from quantum_prior.clustering import decompose_joints
            groups = decompose_joints(C)
            print(f"  Quantum decomposition groups (q=0): {groups}")
        policy = QuantumDecomposedPolicy(
            obs_dim, act_dim, default_groups=groups, hidden=hidden_dim,
        ).to(device)
    else:
        policy = VanillaPolicy(obs_dim, act_dim, hidden=hidden_dim).to(device)

    value_net = ValueNet(obs_dim, hidden=hidden_dim).to(device)

    opt_policy = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.lr_policy)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=ppo_cfg.lr_value)

    buffer = RolloutBuffer(ppo_cfg.n_steps, ppo_cfg.n_envs, obs_dim, act_dim)

    n_updates = ppo_cfg.total_timesteps // (ppo_cfg.n_steps * ppo_cfg.n_envs)
    total_steps = 0
    history = []

    save_dir = os.path.join(
        cfg.get("save_dir", "results/dualarm"), f"{variant}_seed{seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Training {variant} PPO on DualArm (seed={seed}) ===")
    print(f"  {n_updates} updates, {ppo_cfg.total_timesteps} total steps")
    print(f"  reward_mode={reward_mode}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  object_mass={object_mass}, coupling_lambda={coupling_lambda}")
    t0 = time.time()

    for update in range(n_updates):
        t_rollout = time.time()
        rollout_metrics = collect_rollout(
            envs, policy, value_net, buffer, ppo_cfg,
            variant=variant, quantum_computer=quantum_computer,
        )
        dt_rollout = time.time() - t_rollout
        total_steps += ppo_cfg.n_steps * ppo_cfg.n_envs

        t_ppo = time.time()
        policy.train()
        value_net.train()
        update_metrics = ppo_update(
            policy, value_net, buffer, ppo_cfg,
            opt_policy, opt_value, device,
        )
        dt_ppo = time.time() - t_ppo

        log = {
            "step": total_steps,
            **rollout_metrics,
            **update_metrics,
        }
        history.append(log)

        if (update + 1) % 10 == 0 or update < 3:
            elapsed = time.time() - t0
            sps = total_steps / elapsed
            cache_info = ""
            if quantum_computer is not None:
                ci = quantum_computer.cache_info
                cache_info = f" | cache_hit={ci.get('hit_rate', 0):.1%}"
            print(
                f"  Update {update+1}/{n_updates} | "
                f"steps={total_steps} | "
                f"reward={rollout_metrics['mean_reward']:.1f} | "
                f"success={rollout_metrics['success_rate']:.2f} | "
                f"SPS={sps:.0f} | "
                f"rollout={dt_rollout:.1f}s ppo={dt_ppo:.1f}s | "
                f"time={elapsed:.0f}s{cache_info}"
            )

    # Save
    torch.save(policy.state_dict(), os.path.join(save_dir, "policy.pt"))
    torch.save(value_net.state_dict(), os.path.join(save_dir, "value.pt"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if quantum_computer is not None:
        with open(os.path.join(save_dir, "cache_info.json"), "w") as f:
            json.dump(quantum_computer.cache_info, f, indent=2)

    for env in envs:
        env.close()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({total_steps/elapsed:.0f} SPS). Saved to {save_dir}")
    return {"variant": variant, "seed": seed, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--variant", type=str, default="vanilla",
                        choices=ALL_VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(variant=args.variant, config_path=args.config, seed=args.seed)


if __name__ == "__main__":
    main()
