"""GPU training loop for toy dual-arm modal action validation.

All data stays on GPU throughout training. Reuses existing ppo_update().

Usage:
    python -m coupling_rl.train_toy --variant modal_action --seed 0
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import time

import torch
import torch.nn as nn
import yaml

from coupling_rl.networks import VanillaPolicy, ValueNet
from coupling_rl.networks_modal import CouplingFeaturesPolicy, ModalActionPolicy
from coupling_rl.ppo import PPOConfig, ppo_update

ALL_VARIANTS = ["vanilla", "coupling_features", "modal_action"]


# ---------------------------------------------------------------
# GPU Rollout Buffer
# ---------------------------------------------------------------

class GPURolloutBuffer:
    """Rollout buffer with all tensors on GPU."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
        device: str = "cuda",
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs = torch.zeros(n_steps, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, n_envs, act_dim, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        self.advantages = torch.zeros(n_steps, n_envs, device=device)
        self.returns = torch.zeros(n_steps, n_envs, device=device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done.float()
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """GAE on GPU."""
        gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield mini-batches (all GPU tensors)."""
        total = self.n_steps * self.n_envs
        indices = torch.randperm(total, device=self.device)

        obs_flat = self.obs.reshape(total, -1)
        act_flat = self.actions.reshape(total, -1)
        logp_flat = self.log_probs.reshape(total)
        adv_flat = self.advantages.reshape(total)
        ret_flat = self.returns.reshape(total)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield obs_flat[idx], act_flat[idx], logp_flat[idx], adv_flat[idx], ret_flat[idx]

    def reset(self) -> None:
        self.ptr = 0


# ---------------------------------------------------------------
# Collect rollout on GPU
# ---------------------------------------------------------------

@torch.no_grad()
def collect_rollout(
    env,
    policy: nn.Module,
    value_net: nn.Module,
    buffer: GPURolloutBuffer,
    variant: str,
    action_scale: float,
    coupling_info: dict | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> dict[str, float]:
    """Collect one rollout entirely on GPU.

    For modal_action variant, maps 6-dim raw action to 4-dim torque.
    For coupling_features variant, augments obs with J_cross_flat.
    """
    buffer.reset()
    policy.eval()
    value_net.eval()

    obs = env.current_obs  # (n_envs, 13), already on GPU
    ep_rewards = []
    ep_rmses = []
    running_reward = torch.zeros(env.n_envs, device=env.device)
    running_rmse = torch.zeros(env.n_envs, device=env.device)
    running_steps = torch.zeros(env.n_envs, device=env.device)

    for step in range(buffer.n_steps):
        # Build policy input
        if variant == "coupling_features" and coupling_info is not None:
            policy_obs = torch.cat([obs, coupling_info["J_cross_flat"]], dim=-1)
        else:
            policy_obs = obs

        # Get action
        raw_action, log_prob = policy.get_action(policy_obs)
        value = value_net(policy_obs)

        # Map action to env torques
        if variant == "modal_action" and coupling_info is not None:
            env_action = ModalActionPolicy.map_to_torques(
                raw_action,
                coupling_info["U"],
                coupling_info["sigma"],
                coupling_info["Vh"],
                action_scale,
            )
            # env.step expects [-1,1] range, undo scale for env clamping
            env_action = env_action / action_scale
        else:
            env_action = raw_action

        obs_next, reward, done, info = env.step(env_action)

        buffer.add(policy_obs, raw_action, reward, done, log_prob, value)

        # Track episode stats
        running_reward += reward
        running_rmse += info["rmse"]
        running_steps += 1

        if done.any():
            mask = done
            ep_rewards.append(running_reward[mask].mean().item())
            ep_rmses.append((running_rmse[mask] / running_steps[mask]).mean().item())
            running_reward[mask] = 0.0
            running_rmse[mask] = 0.0
            running_steps[mask] = 0.0

        obs = obs_next

    # Last value for GAE
    if variant == "coupling_features" and coupling_info is not None:
        last_obs = torch.cat([obs, coupling_info["J_cross_flat"]], dim=-1)
    else:
        last_obs = obs
    last_value = value_net(last_obs)
    buffer.compute_gae(last_value, gamma=gamma, lam=gae_lambda)

    return {
        "mean_reward": float(sum(ep_rewards) / max(len(ep_rewards), 1)),
        "mean_rmse": float(sum(ep_rmses) / max(len(ep_rmses), 1)),
        "n_episodes": len(ep_rewards),
    }


# ---------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------

def train_toy(
    variant: str = "vanilla",
    seed: int = 0,
    config: dict | None = None,
    device: str = "cuda",
    save_dir: str = "results/toy",
) -> dict:
    """Train one variant+seed. Returns history dict."""
    from tqdm import tqdm
    from envs.toy_dualarm_vec import ToyDualArmVecEnv

    if config is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "toy.yaml")
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

    torch.manual_seed(seed)

    # Environment
    env = ToyDualArmVecEnv(
        n_envs=config.get("n_envs", 512),
        n_per_arm=config.get("n_per_arm", 2),
        object_mass=config.get("object_mass", 0.5),
        link_masses=tuple(config.get("arm_link_mass", [1.0, 0.5])),
        link_lengths=tuple(config.get("arm_link_length", [0.3, 0.25])),
        damping=config.get("damping", 0.5),
        dt=config.get("dt", 0.02),
        max_steps=config.get("max_steps", 200),
        traj_freq_range=tuple(config.get("trajectory_freq", [0.5, 0.8])),
        traj_amp_range=tuple(config.get("trajectory_amp", [0.3, 0.6])),
        action_scale=config.get("action_scale", 10.0),
        device=device,
    )
    obs = env.reset(seed=seed)

    # Dimensions
    hidden = config.get("hidden_dim", 128)
    action_scale = config.get("action_scale", 10.0)

    if variant == "vanilla":
        obs_dim, act_dim = 13, 4
        policy = VanillaPolicy(obs_dim, act_dim, hidden).to(device)
    elif variant == "coupling_features":
        obs_dim, act_dim = 17, 4
        policy = CouplingFeaturesPolicy(obs_dim, act_dim, hidden).to(device)
    elif variant == "modal_action":
        obs_dim, act_dim = 13, ModalActionPolicy.RAW_DIM  # 6
        policy = ModalActionPolicy(13, env.n_per_arm, hidden).to(device)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    value_obs_dim = obs_dim
    value_net = ValueNet(value_obs_dim, hidden).to(device)

    # PPO config
    n_steps = config.get("n_steps", 256)
    ppo_cfg = PPOConfig(
        clip_eps=config.get("clip_eps", 0.2),
        lr_policy=config.get("lr_policy", 3e-4),
        lr_value=config.get("lr_value", 1e-3),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        entropy_coef=config.get("entropy_coef", 0.01),
        n_epochs=config.get("ppo_epochs", 4),
        mini_batch_size=config.get("mini_batch_size", 4096),
        n_steps=n_steps,
        n_envs=config.get("n_envs", 512),
        total_timesteps=config.get("total_timesteps", 500_000),
        action_scale=action_scale,
    )

    opt_policy = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.lr_policy)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=ppo_cfg.lr_value)
    buffer = GPURolloutBuffer(n_steps, ppo_cfg.n_envs, obs_dim, act_dim, device)

    total_timesteps = ppo_cfg.total_timesteps
    steps_per_update = n_steps * ppo_cfg.n_envs
    n_updates = total_timesteps // steps_per_update

    # Training loop
    history = []
    total_steps = 0
    run_dir = os.path.join(save_dir, f"{variant}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    pbar = tqdm(total=total_timesteps, desc=f"{variant}/s{seed}", unit="step")

    for update in range(n_updates):
        # Update coupling info once per rollout (not per step)
        coupling_info = env.get_coupling_info()

        rollout_metrics = collect_rollout(
            env, policy, value_net, buffer, variant, action_scale, coupling_info,
            gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda,
        )
        total_steps += steps_per_update

        # PPO update (reuses existing function — works because GPU buffer
        # yields GPU tensors and policy/value_net are on GPU)
        policy.train()
        value_net.train()
        update_metrics = ppo_update(
            policy, value_net, buffer, ppo_cfg, opt_policy, opt_value, device,
        )

        log = {"step": total_steps, **rollout_metrics, **update_metrics}
        history.append(log)

        pbar.update(steps_per_update)
        pbar.set_postfix(
            rmse=f"{rollout_metrics['mean_rmse']:.3f}",
            rew=f"{rollout_metrics['mean_reward']:.1f}",
        )

    pbar.close()

    # Save
    torch.save(policy.state_dict(), os.path.join(run_dir, "policy.pt"))
    torch.save(value_net.state_dict(), os.path.join(run_dir, "value.pt"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Saved to {run_dir}")
    return {"variant": variant, "seed": seed, "history": history}


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train toy dual-arm PPO variant")
    parser.add_argument("--variant", type=str, default="vanilla", choices=ALL_VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_toy(variant=args.variant, seed=args.seed, config=cfg, device=device)


if __name__ == "__main__":
    main()
