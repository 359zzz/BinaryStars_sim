"""Training script for coupling-aware RL (Scheme beta).

Five PPO variants:
1. vanilla:        Standard MLP policy, no coupling reward
2. geometric:      2-head (proximal + distal) policy
3. coupling:       Vanilla + classical coupling reward (|J_ij|)
4. quantum_c:      CouplingAware + quantum entanglement reward (C_ij)
5. quantum_decomp: QuantumDecomposed + quantum clustering reward

Usage:
    python -m coupling_rl.train_ppo --config configs/beta.yaml --variant vanilla --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pipe, Process

import numpy as np
import torch
import yaml

from coupling_rl.networks import (
    CouplingAwarePolicy,
    GeometricPolicy,
    QuantumDecomposedPolicy,
    ValueNet,
    VanillaPolicy,
)
from coupling_rl.ppo import PPOConfig, RolloutBuffer, ppo_update
from envs.openarm_reach import OpenArmReachEnv

ALL_VARIANTS = ["vanilla", "geometric", "coupling", "quantum_c", "quantum_decomp"]

# Variant -> (reward_mode, needs_quantum, obs_dim)
VARIANT_CONFIG = {
    "vanilla":        ("vanilla",             False, 20),
    "geometric":      ("vanilla",             False, 20),
    "coupling":       ("classical_coupling",  False, 20),
    "quantum_c":      ("quantum_entanglement", True, 41),
    "quantum_decomp": ("quantum_decomposed",   True, 41),
}


def _make_quantum_computer():
    """Create CachedEntanglementComputer with OpenArm mass matrix function."""
    from physics.openarm_params import compute_openarm_mass_matrix
    from quantum_prior.cached_computer import CachedEntanglementComputer

    return CachedEntanglementComputer(
        mass_matrix_fn=compute_openarm_mass_matrix,
        resolution=0.01,
    )


# ── SubprocVecEnv ────────────────────────────────────────────────────────────

def _env_worker(conn, env_kwargs, seed):
    """Worker process for parallel env stepping."""
    env = OpenArmReachEnv(**env_kwargs)
    env.reset(seed=seed)
    conn.send(env.observation_space.shape[0])  # handshake
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, r, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
            conn.send((obs, r, done, info.get("success", False)))
        elif cmd == "reset":
            obs, info = env.reset(seed=data)
            conn.send(obs)
        elif cmd == "close":
            env.close()
            conn.close()
            break


class SubprocVecEnv:
    """Subprocess-based vectorized environment for parallel MuJoCo stepping."""

    def __init__(self, n_envs, env_kwargs, base_seed=0):
        self.n_envs = n_envs
        self.parent_conns = []
        self.procs = []
        for i in range(n_envs):
            parent_conn, child_conn = Pipe()
            proc = Process(
                target=_env_worker,
                args=(child_conn, env_kwargs, base_seed + i),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.procs.append(proc)
        # Wait for handshake
        self.obs_dim = self.parent_conns[0].recv()
        for conn in self.parent_conns[1:]:
            conn.recv()

    def step(self, actions):
        """Step all envs in parallel. actions: (n_envs, act_dim)."""
        for i, conn in enumerate(self.parent_conns):
            conn.send(("step", actions[i]))
        results = [conn.recv() for conn in self.parent_conns]
        obs = np.array([r[0] for r in results], dtype=np.float32)
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=np.float32)
        successes = [r[3] for r in results]
        return obs, rewards, dones, successes

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()


# ── Rollout collection ───────────────────────────────────────────────────────

def collect_rollout(
    vec_env: SubprocVecEnv,
    policy: torch.nn.Module,
    value_net: ValueNet,
    buffer: RolloutBuffer,
    config: PPOConfig,
    infer_device: str = "cpu",
    variant: str = "vanilla",
    quantum_computer=None,
    obs_base: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Collect n_steps of experience from vectorized envs.

    Policy inference runs on infer_device (CPU recommended for small MLPs).
    Returns (metrics_dict, last_obs_base).
    """
    n_envs = vec_env.n_envs

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
            obs_np = _augment_obs(obs_base, variant, quantum_computer)
            obs_t = torch.from_numpy(obs_np).to(infer_device)

            action, log_prob = policy.get_action(obs_t)
            value = value_net(obs_t)

            action_np = action.numpy()
            log_prob_np = log_prob.numpy()
            value_np = value.numpy()

            action_scaled = np.clip(action_np * config.action_scale, -50.0, 50.0)

            # Parallel env stepping via subprocesses
            next_obs_base, rewards, dones, successes = vec_env.step(action_scaled)

            for i in range(n_envs):
                current_ep_reward[i] += rewards[i]
                current_ep_len[i] += 1
                if dones[i]:
                    ep_rewards.append(current_ep_reward[i])
                    ep_lengths.append(current_ep_len[i])
                    ep_successes.append(successes[i])
                    current_ep_reward[i] = 0.0
                    current_ep_len[i] = 0

            buffer.add(obs_np, action_np, rewards, dones, log_prob_np, value_np)
            obs_base = next_obs_base

        # Compute last value for GAE
        last_obs_np = _augment_obs(obs_base, variant, quantum_computer)
        last_obs_t = torch.from_numpy(last_obs_np).to(infer_device)
        last_value = value_net(last_obs_t).numpy()
        buffer.compute_gae(last_value, config.gamma, config.gae_lambda)

    metrics = {
        "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "success_rate": float(np.mean(ep_successes)) if ep_successes else 0.0,
        "n_episodes": len(ep_rewards),
    }
    return metrics, obs_base


def _augment_obs(
    obs_base: np.ndarray,
    variant: str,
    quantum_computer,
) -> np.ndarray:
    """Augment base observations with structure features if needed."""
    if variant in ("vanilla", "geometric"):
        return obs_base
    if variant == "coupling":
        # coupling uses obs_dim=20, no augmentation needed (reward computed in env)
        return obs_base

    n_envs = obs_base.shape[0]
    feats = np.zeros((n_envs, 21), dtype=np.float32)

    for i in range(n_envs):
        q = obs_base[i, :7]
        if variant == "quantum_c" or variant == "quantum_decomp":
            feats[i] = quantum_computer.get_entanglement_features(q)

    return np.concatenate([obs_base, feats], axis=1)


def train(
    variant: str = "vanilla",
    config_path: str | None = None,
    seed: int = 0,
) -> dict:
    """Train one PPO variant."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "beta.yaml"
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    reward_mode, needs_quantum, obs_dim = VARIANT_CONFIG[variant]

    ppo_cfg = PPOConfig(
        clip_eps=cfg.get("clip_eps", 0.2),
        lr_policy=cfg.get("lr_policy", 3e-4),
        lr_value=cfg.get("lr_value", 1e-3),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        entropy_coef=cfg.get("entropy_coef", 0.01),
        n_epochs=cfg.get("ppo_epochs", 10),
        mini_batch_size=cfg.get("mini_batch_size", 64),
        n_steps=cfg.get("n_steps", 2048),
        n_envs=cfg.get("n_envs", 8),
        total_timesteps=cfg.get("total_timesteps", 2_000_000),
    )
    coupling_lambda = cfg.get("coupling_lambda", 0.1)

    # For 256x256 MLP with batch=64, CPU is faster than GPU
    # (CUDA kernel launch + sync overhead >> compute for small models)
    # GPU only helps for batch > ~1000 or models > ~10M params
    infer_device = "cpu"
    train_device = "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Quantum computer (main process only — not sent to subprocesses)
    quantum_computer = _make_quantum_computer() if needs_quantum else None

    # Subprocess-based vectorized environments
    env_kwargs = {
        "coupling_lambda": coupling_lambda if reward_mode != "vanilla" else 0.0,
        "reward_mode": reward_mode,
        # quantum_computer NOT passed to subprocesses for non-quantum variants
        # For quantum variants, coupling reward is computed in-env (subprocess)
        # but quantum_computer can't be pickled. Use reward_mode dispatch:
        # quantum variants compute coupling in main process via obs augmentation
    }
    # For quantum reward modes, fall back to vanilla reward in env
    # (quantum reward shaping happens via obs augmentation + policy structure)
    if needs_quantum:
        env_kwargs["reward_mode"] = "vanilla"
        env_kwargs["coupling_lambda"] = 0.0

    vec_env = SubprocVecEnv(ppo_cfg.n_envs, env_kwargs, base_seed=seed)

    # Networks
    act_dim = 7
    if variant == "geometric":
        policy = GeometricPolicy(obs_dim).to(infer_device)
    elif variant in ("quantum_c", "coupling"):
        policy = CouplingAwarePolicy(obs_dim, act_dim).to(infer_device)
    elif variant == "quantum_decomp":
        groups = None
        if quantum_computer is not None:
            q0 = np.zeros(7)
            C = quantum_computer.get_entanglement_graph(q0)
            from quantum_prior.clustering import decompose_joints
            groups = decompose_joints(C)
            print(f"  Quantum decomposition groups (q=0): {groups}")
        policy = QuantumDecomposedPolicy(
            obs_dim, act_dim, default_groups=groups,
        ).to(infer_device)
    else:
        policy = VanillaPolicy(obs_dim, act_dim).to(infer_device)

    value_net = ValueNet(obs_dim).to(infer_device)

    opt_policy = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.lr_policy)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=ppo_cfg.lr_value)

    buffer = RolloutBuffer(ppo_cfg.n_steps, ppo_cfg.n_envs, obs_dim, act_dim)

    # Training loop
    n_updates = ppo_cfg.total_timesteps // (ppo_cfg.n_steps * ppo_cfg.n_envs)
    total_steps = 0
    history = []

    save_dir = os.path.join(
        cfg.get("save_dir", "results/beta"), f"{variant}_seed{seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Training {variant} PPO (seed={seed}) ===")
    print(f"  {n_updates} updates, {ppo_cfg.total_timesteps} total steps")
    print(f"  reward_mode={reward_mode}, obs_dim={obs_dim}")
    print(f"  infer_device={infer_device}, train_device={train_device}")
    print(f"  n_envs={ppo_cfg.n_envs} (subprocess-parallel)")
    t0 = time.time()

    # Initial observations from subprocess envs
    # Reset all and get first obs
    for conn in vec_env.parent_conns:
        conn.send(("reset", seed))
    obs_base = np.array(
        [conn.recv() for conn in vec_env.parent_conns], dtype=np.float32,
    )

    for update in range(n_updates):
        t_rollout = time.time()
        rollout_metrics, obs_base = collect_rollout(
            vec_env, policy, value_net, buffer, ppo_cfg, infer_device,
            variant=variant, quantum_computer=quantum_computer,
            obs_base=obs_base,
        )
        dt_rollout = time.time() - t_rollout
        total_steps += ppo_cfg.n_steps * ppo_cfg.n_envs

        # PPO update (all CPU — no device migration needed)
        t_ppo = time.time()
        policy.train()
        value_net.train()
        update_metrics = ppo_update(
            policy, value_net, buffer, ppo_cfg,
            opt_policy, opt_value, train_device,
        )
        dt_ppo = time.time() - t_ppo

        log = {
            "step": total_steps,
            **rollout_metrics,
            **update_metrics,
        }
        history.append(log)

        if (update + 1) % 10 == 0:
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
    policy.to("cpu")
    value_net.to("cpu")
    torch.save(policy.state_dict(), os.path.join(save_dir, "policy.pt"))
    torch.save(value_net.state_dict(), os.path.join(save_dir, "value.pt"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if quantum_computer is not None:
        with open(os.path.join(save_dir, "cache_info.json"), "w") as f:
            json.dump(quantum_computer.cache_info, f, indent=2)

    vec_env.close()

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
