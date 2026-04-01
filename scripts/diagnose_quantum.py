"""Diagnose where quantum_c training hangs.

Tests each component in isolation with timing.
Run: python scripts/diagnose_quantum.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

def timed(label):
    """Context manager that prints timing."""
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"  [{label}] starting...", end="", flush=True)
            return self
        def __exit__(self, *args):
            dt = time.time() - self.t0
            print(f" {dt:.3f}s", flush=True)
    return Timer()

print("=" * 60, flush=True)
print("Quantum_c training hang diagnosis", flush=True)
print("=" * 60, flush=True)

# Step 1: numpy basics
with timed("numpy eigh 7x7"):
    import numpy as np
    A = np.random.randn(7, 7)
    A = A + A.T
    for _ in range(1000):
        np.linalg.eigh(A)

# Step 2: torch basics
with timed("torch linear 8x20 -> 256"):
    import torch
    torch.set_num_threads(2)
    x = torch.randn(8, 20)
    m = torch.nn.Linear(20, 256)
    for _ in range(1000):
        m(x)

# Step 3: mass matrix
with timed("compute_openarm_mass_matrix x100"):
    from physics.openarm_params import compute_openarm_mass_matrix
    q = np.zeros(7)
    for _ in range(100):
        M = compute_openarm_mass_matrix(q)

# Step 4: entanglement graph (single call)
with timed("compute_entanglement_graph x1"):
    from quantum_prior.entanglement_graph import compute_entanglement_graph
    C = compute_entanglement_graph(M, t_max=3.0, n_time_steps=50)
    print(f"\n    C shape={C.shape}, max={C.max():.4f}", end="", flush=True)

# Step 5: cached computer
with timed("CachedEntanglementComputer create"):
    from quantum_prior.cached_computer import CachedEntanglementComputer
    qc = CachedEntanglementComputer(
        mass_matrix_fn=compute_openarm_mass_matrix,
        resolution=0.01,
    )

with timed("cached get_entanglement_graph x100"):
    for _ in range(100):
        qc.get_entanglement_graph(q)
    print(f"\n    hit_rate={qc.hit_rate:.2%}", end="", flush=True)

with timed("cached get_entanglement_features x100"):
    for _ in range(100):
        qc.get_entanglement_features(q)

with timed("cached get_classical_coupling x100"):
    for _ in range(100):
        qc.get_classical_coupling(q)

# Step 6: MuJoCo env
with timed("OpenArmReachEnv create (quantum mode)"):
    from envs.openarm_reach import OpenArmReachEnv
    env = OpenArmReachEnv(
        coupling_lambda=0.1,
        reward_mode="quantum_entanglement",
        quantum_computer=qc,
    )

with timed("env.reset()"):
    obs, info = env.reset(seed=0)

with timed("env.step() x200 (1 episode)"):
    action = np.zeros(7, dtype=np.float32)
    for i in range(200):
        action = np.random.uniform(-5, 5, size=7).astype(np.float32)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()
    print(f"\n    cache hit_rate={qc.hit_rate:.2%}", end="", flush=True)

# Step 7: policy + env together (mimics collect_rollout)
with timed("VanillaPolicy create"):
    from coupling_rl.networks import CouplingAwarePolicy, ValueNet
    policy = CouplingAwarePolicy(41, 7)
    value_net = ValueNet(41)
    policy.eval()
    value_net.eval()

with timed("policy forward x100"):
    obs_t = torch.randn(8, 41)
    with torch.no_grad():
        for _ in range(100):
            policy.get_action(obs_t)

with timed("full collect_rollout simulation (50 steps x 1 env)"):
    env2 = OpenArmReachEnv(
        coupling_lambda=0.1,
        reward_mode="quantum_entanglement",
        quantum_computer=qc,
    )
    obs, _ = env2.reset(seed=0)
    with torch.no_grad():
        for step in range(50):
            q = obs[:7]
            feats = qc.get_entanglement_features(q)
            obs_aug = np.concatenate([obs, feats])
            obs_t = torch.from_numpy(obs_aug).unsqueeze(0)
            action, lp = policy.get_action(obs_t)
            value = value_net(obs_t)
            action_np = np.clip(action.numpy()[0] * 50.0, -50, 50)
            obs, r, term, trunc, info = env2.step(action_np)
            if term or trunc:
                obs, _ = env2.reset()
            if step % 10 == 0:
                print(f"\n    step {step}/50 ok", end="", flush=True)

# Step 8: actual collect_rollout with 8 envs
print("\n", flush=True)
with timed("make_envs (8 envs, quantum_entanglement)"):
    from coupling_rl.train_ppo import make_envs
    envs = make_envs(8, "quantum_c", 0.1, 0, quantum_computer=qc)

with timed("collect_rollout (n_steps=100, 8 envs)"):
    from coupling_rl.ppo import PPOConfig, RolloutBuffer
    from coupling_rl.train_ppo import collect_rollout
    cfg = PPOConfig(n_steps=100, n_envs=8, mini_batch_size=64)
    buf = RolloutBuffer(100, 8, 41, 7)
    metrics = collect_rollout(envs, policy, value_net, buf, cfg,
                              variant="quantum_c", quantum_computer=qc)
    print(f"\n    reward={metrics['mean_reward']:.1f}", end="", flush=True)

print("\n\n=== All tests passed! Training should work. ===", flush=True)
