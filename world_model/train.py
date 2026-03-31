"""Training script for world models (Scheme gamma).

Five models:
1. MLP:   [q, dq, tau] = 21-dim -> base MLP ensemble
2. J-MLP: [q, dq, tau, |J|] = 42-dim -> classical prior
3. C-MLP: [q, dq, tau, C] = 42-dim -> quantum prior
4. DeLaN: [q, dq, tau] = 21-dim -> Lagrangian structure
5. CRBA:  physics-based (no training needed)

Usage:
    python -m world_model.train --config configs/gamma.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from world_model.data_collector import collect_all_payloads
from world_model.dataset import (
    QuantumTransitionDataset,
    TransitionDataset,
)
from world_model.mlp_ensemble import (
    MLPEnsemble,
    QuantumMLPEnsemble,
    train_ensemble,
)
from world_model.delan import DeLaN, train_delan


def _make_feature_fns():
    """Create classical and quantum feature functions."""
    from physics.openarm_params import compute_openarm_mass_matrix
    from quantum_prior.cached_computer import CachedEntanglementComputer

    qc = CachedEntanglementComputer(
        mass_matrix_fn=compute_openarm_mass_matrix, resolution=0.01,
    )

    def classical_features(q):
        return qc.get_classical_features(q)

    def quantum_features(q):
        return qc.get_entanglement_features(q)

    return classical_features, quantum_features


def main(config_path: str | None = None) -> None:
    # Load config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "gamma.yaml"
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    save_dir = cfg.get("save_dir", "results/gamma")
    os.makedirs(save_dir, exist_ok=True)
    data_dir = os.path.join(save_dir, "data")

    # Which models to train
    models_to_train = cfg.get(
        "models", ["mlp", "j_mlp", "c_mlp", "delan"]
    )

    # ── Step 1: Collect data ─────────────────────────────────────────────
    payloads = cfg.get("payloads", [0.0, 0.5, 1.0])
    n_rollouts = cfg.get("n_rollouts", 200)
    rollout_len = cfg.get("rollout_len", 100)
    seed = cfg.get("seed", 42)

    print("\n=== Collecting transition data ===")
    all_data = collect_all_payloads(
        payloads=payloads,
        n_rollouts=n_rollouts,
        rollout_len=rollout_len,
        seed=seed,
        save_dir=data_dir,
    )

    # ── Step 2: Train on source domain (0 kg) ────────────────────────────
    train_payload = cfg.get("train_payload", 0.0)
    train_data = all_data[train_payload]
    batch_size = cfg.get("batch_size", 256)

    print(f"\n=== Training on payload={train_payload} kg ({train_data['q'].shape[0]} samples) ===")

    meta = {"train_payload": train_payload, "n_train": train_data["q"].shape[0]}

    # ── MLP (base) ────────────────────────────────────────────────────────
    if "mlp" in models_to_train:
        print("\n--- MLP Ensemble (base) ---")
        train_dataset = TransitionDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        mlp = MLPEnsemble(
            n_models=cfg.get("ensemble_size", 5),
            hidden=cfg.get("mlp_hidden", 256),
        )
        t0 = time.time()
        mlp_losses = train_ensemble(
            mlp, train_loader, epochs=cfg.get("mlp_epochs", 100),
            lr=cfg.get("lr", 1e-3), device=device,
        )
        mlp_time = time.time() - t0
        torch.save(mlp.state_dict(), os.path.join(save_dir, "mlp_ensemble.pt"))
        meta["mlp_time_s"] = mlp_time
        meta["mlp_final_loss"] = mlp_losses[-1]
        print(f"  MLP training: {mlp_time:.1f}s, final loss={mlp_losses[-1]:.6f}")

    # ── J-MLP (classical features) ────────────────────────────────────────
    if "j_mlp" in models_to_train:
        print("\n--- J-MLP Ensemble (classical |J_ij| features) ---")
        classical_fn, _ = _make_feature_fns()
        t0 = time.time()
        j_dataset = QuantumTransitionDataset(train_data, classical_fn)
        print(f"  Feature precomputation: {time.time() - t0:.1f}s")
        j_loader = DataLoader(j_dataset, batch_size=batch_size, shuffle=True)
        j_mlp = QuantumMLPEnsemble(
            n_models=cfg.get("ensemble_size", 5),
            hidden=cfg.get("mlp_hidden", 256),
        )
        t0 = time.time()
        j_losses = train_ensemble(
            j_mlp, j_loader, epochs=cfg.get("mlp_epochs", 100),
            lr=cfg.get("lr", 1e-3), device=device,
        )
        j_time = time.time() - t0
        torch.save(j_mlp.state_dict(), os.path.join(save_dir, "j_mlp_ensemble.pt"))
        meta["j_mlp_time_s"] = j_time
        meta["j_mlp_final_loss"] = j_losses[-1]
        print(f"  J-MLP training: {j_time:.1f}s, final loss={j_losses[-1]:.6f}")

    # ── C-MLP (quantum features) ──────────────────────────────────────────
    if "c_mlp" in models_to_train:
        print("\n--- C-MLP Ensemble (quantum C_ij features) ---")
        _, quantum_fn = _make_feature_fns()
        t0 = time.time()
        c_dataset = QuantumTransitionDataset(train_data, quantum_fn)
        precomp_time = time.time() - t0
        print(f"  Feature precomputation: {precomp_time:.1f}s")
        c_loader = DataLoader(c_dataset, batch_size=batch_size, shuffle=True)
        c_mlp = QuantumMLPEnsemble(
            n_models=cfg.get("ensemble_size", 5),
            hidden=cfg.get("mlp_hidden", 256),
        )
        t0 = time.time()
        c_losses = train_ensemble(
            c_mlp, c_loader, epochs=cfg.get("mlp_epochs", 100),
            lr=cfg.get("lr", 1e-3), device=device,
        )
        c_time = time.time() - t0
        torch.save(c_mlp.state_dict(), os.path.join(save_dir, "c_mlp_ensemble.pt"))
        meta["c_mlp_time_s"] = c_time
        meta["c_mlp_final_loss"] = c_losses[-1]
        meta["c_mlp_precompute_s"] = precomp_time
        print(f"  C-MLP training: {c_time:.1f}s, final loss={c_losses[-1]:.6f}")

    # ── DeLaN ─────────────────────────────────────────────────────────────
    if "delan" in models_to_train:
        print("\n--- DeLaN ---")
        train_dataset_base = TransitionDataset(train_data)
        base_loader = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True)
        delan = DeLaN(hidden=cfg.get("delan_hidden", 128))
        t0 = time.time()
        delan_losses = train_delan(
            delan, base_loader, epochs=cfg.get("delan_epochs", 200),
            lr=cfg.get("lr", 1e-3), device=device,
        )
        delan_time = time.time() - t0
        torch.save(delan.state_dict(), os.path.join(save_dir, "delan.pt"))
        meta["delan_time_s"] = delan_time
        meta["delan_final_loss"] = delan_losses[-1]
        print(f"  DeLaN training: {delan_time:.1f}s, final loss={delan_losses[-1]:.6f}")

    # Save training metadata
    with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Training complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
