"""Evaluate world model transfer across payloads (Scheme gamma).

Protocol:
1. Models trained on 0 kg data
2. Evaluate on {0, 0.5, 1.0} kg test sets
3. Fine-tune sweep: N = {0, 10, 50, 100, 500, 1000, 5000}
4. Metrics: 1-step RMSE + multi-step rollout RMSE
5. Special: indirect_coupling_rmse on 8 classically-decoupled pairs

Five models: MLP, J-MLP, C-MLP, DeLaN, CRBA

Usage:
    python -m world_model.evaluate_transfer --config configs/gamma.yaml
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

from world_model.crba_model import CRBAModel
from world_model.dataset import (
    QuantumTransitionDataset,
    TransitionDataset,
    load_npz,
)
from world_model.delan import DeLaN, train_delan
from world_model.mlp_ensemble import (
    MLPEnsemble,
    QuantumMLPEnsemble,
    train_ensemble,
)

# Indirect coupling pairs: |J_ij| < 0.01 but C_ij > 0.05 (from Exp A')
# These 7 pairs are the "smoking gun" for quantum advantage
INDIRECT_COUPLING_PAIRS = [
    (0, 1), (0, 2), (1, 2), (1, 5), (2, 5), (2, 6), (5, 6),
]


def one_step_rmse(
    pred: dict[str, np.ndarray], truth: dict[str, np.ndarray]
) -> dict[str, float]:
    """Compute 1-step RMSE for q and dq."""
    q_err = np.sqrt(np.mean((pred["q_next"] - truth["q_next"])**2))
    dq_err = np.sqrt(np.mean((pred["dq_next"] - truth["dq_next"])**2))
    total = np.sqrt(np.mean(
        np.concatenate([
            (pred["q_next"] - truth["q_next"]).ravel(),
            (pred["dq_next"] - truth["dq_next"]).ravel(),
        ])**2
    ))
    return {"q_rmse": float(q_err), "dq_rmse": float(dq_err), "total_rmse": float(total)}


def indirect_coupling_rmse(
    pred: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    pairs: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Compute acceleration RMSE specifically for indirect coupling joint pairs.

    These are pairs where J_ij ~ 0 (classically decoupled) but C_ij > 0
    (quantum entangled via shared ancestors). A model that understands quantum
    structure should predict these accurately; classical models fail here.

    Parameters
    ----------
    pred, truth : dicts with 'dq_next' arrays (N, 7)
    pairs : list of (i, j) pairs to evaluate. Defaults to INDIRECT_COUPLING_PAIRS.

    Returns
    -------
    dict with per-pair and mean RMSE
    """
    if pairs is None:
        pairs = INDIRECT_COUPLING_PAIRS

    ddq_pred = pred["dq_next"]  # (N, 7)
    ddq_true = truth["dq_next"]

    results = {}
    pair_rmses = []
    for i, j in pairs:
        # Joint-pair acceleration error
        err_i = ddq_pred[:, i] - ddq_true[:, i]
        err_j = ddq_pred[:, j] - ddq_true[:, j]
        rmse = float(np.sqrt(np.mean(err_i**2 + err_j**2)))
        results[f"pair_{i}_{j}_rmse"] = rmse
        pair_rmses.append(rmse)

    results["mean_indirect_rmse"] = float(np.mean(pair_rmses))
    return results


def multi_step_rmse_crba(
    crba: CRBAModel,
    data: dict[str, np.ndarray],
    horizon: int = 50,
) -> float:
    """Multi-step rollout RMSE for CRBA model."""
    n = data["q"].shape[0]
    n_rollouts = min(n // horizon, 100)
    errors = []

    for r in range(n_rollouts):
        start = r * horizon
        q = data["q"][start].copy()
        dq = data["dq"][start].copy()

        for t in range(horizon):
            idx = start + t
            if idx >= n:
                break
            tau = data["tau"][idx]
            pred = crba.predict_batch({
                "q": q[np.newaxis],
                "dq": dq[np.newaxis],
                "tau": tau[np.newaxis],
            })
            q = pred["q_next"][0]
            dq = pred["dq_next"][0]
            true_q = data["q_next"][idx]
            true_dq = data["dq_next"][idx]
            err = np.concatenate([q - true_q, dq - true_dq])
            errors.append(np.mean(err**2))

    return float(np.sqrt(np.mean(errors)))


def multi_step_rmse_nn(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    horizon: int = 50,
    device: str = "cuda",
    feature_fn=None,
) -> float:
    """Multi-step rollout RMSE for neural models.

    For quantum models (feature_fn is not None), recomputes features at each step.
    """
    model.eval()
    n = data["q"].shape[0]
    n_rollouts = min(n // horizon, 100)
    errors = []

    for r in range(n_rollouts):
        start = r * horizon
        q = torch.from_numpy(data["q"][start:start+1]).float().to(device)
        dq = torch.from_numpy(data["dq"][start:start+1]).float().to(device)

        for t in range(horizon):
            idx = start + t
            if idx >= n:
                break
            tau = torch.from_numpy(data["tau"][idx:idx+1]).float().to(device)

            if feature_fn is not None:
                q_np = q[0].detach().cpu().numpy()
                feats = torch.from_numpy(
                    feature_fn(q_np)[np.newaxis]
                ).float().to(device)
                x = torch.cat([q, dq, tau, feats], dim=-1)
            else:
                x = torch.cat([q, dq, tau], dim=-1)

            x.requires_grad_(True)
            pred = model(x)
            q = pred[:, :7].detach()
            dq = pred[:, 7:].detach()

            true = np.concatenate([data["q_next"][idx], data["dq_next"][idx]])
            pred_np = pred[0].detach().cpu().numpy()
            err = pred_np - true
            errors.append(np.mean(err**2))

    return float(np.sqrt(np.mean(errors)))


def evaluate_model_nn(
    model: torch.nn.Module,
    test_data: dict[str, np.ndarray],
    device: str = "cuda",
    feature_fn=None,
) -> dict[str, np.ndarray]:
    """1-step predictions from a neural model."""
    model.eval()
    base = np.concatenate([test_data["q"], test_data["dq"], test_data["tau"]], axis=1)

    if feature_fn is not None:
        n = base.shape[0]
        features = np.zeros((n, 21), dtype=np.float32)
        for i in range(n):
            features[i] = feature_fn(test_data["q"][i])
        x = np.concatenate([base, features], axis=1)
    else:
        x = base

    x_t = torch.from_numpy(x).float().to(device)
    # DeLaN needs autograd for gravity computation — no torch.no_grad()
    x_t.requires_grad_(True)
    pred = model(x_t)
    pred_np = pred.detach().cpu().numpy()
    return {"q_next": pred_np[:, :7], "dq_next": pred_np[:, 7:]}


def finetune_and_evaluate(
    model_class: str,
    base_state_dict: dict,
    finetune_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    n_samples: int,
    epochs: int = 50,
    lr: float = 5e-4,
    device: str = "cuda",
    feature_fn=None,
    **model_kwargs,
) -> dict:
    """Fine-tune a model on N samples, then evaluate."""
    if model_class == "mlp":
        model = MLPEnsemble(**model_kwargs)
    elif model_class in ("j_mlp", "c_mlp"):
        model = QuantumMLPEnsemble(**model_kwargs)
    else:
        model = DeLaN(**model_kwargs)

    model.load_state_dict(base_state_dict)

    if n_samples > 0:
        if feature_fn is not None:
            ft_dataset = QuantumTransitionDataset(
                finetune_data, feature_fn, max_samples=n_samples,
            )
        else:
            ft_dataset = TransitionDataset(finetune_data, max_samples=n_samples)
        ft_loader = DataLoader(ft_dataset, batch_size=min(64, n_samples), shuffle=True)

        if model_class in ("mlp", "j_mlp", "c_mlp"):
            train_ensemble(model, ft_loader, epochs=epochs, lr=lr, device=device)
        else:
            train_delan(model, ft_loader, epochs=epochs, lr=lr, device=device)

    model = model.to(device)
    pred = evaluate_model_nn(model, test_data, device, feature_fn=feature_fn)
    metrics = one_step_rmse(pred, test_data)
    metrics["multi_step_rmse"] = multi_step_rmse_nn(
        model, test_data, device=device, feature_fn=feature_fn,
    )
    # Indirect coupling RMSE (smoking gun metric)
    indirect_metrics = indirect_coupling_rmse(pred, test_data)
    metrics.update(indirect_metrics)
    return metrics


def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "gamma.yaml"
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = cfg.get("save_dir", "results/gamma")
    data_dir = os.path.join(save_dir, "data")

    payloads = cfg.get("payloads", [0.0, 0.5, 1.0])
    finetune_sizes = cfg.get("finetune_sizes", [0, 10, 50, 100, 500, 1000, 5000])
    n_test = cfg.get("n_test", 2000)
    models_to_eval = cfg.get("models", ["mlp", "j_mlp", "c_mlp", "delan"])

    # Load trained models
    model_registry = {}  # name -> (model, base_state_dict, feature_fn, kwargs)

    if "mlp" in models_to_eval:
        mlp = MLPEnsemble(
            n_models=cfg.get("ensemble_size", 5), hidden=cfg.get("mlp_hidden", 256),
        )
        mlp.load_state_dict(torch.load(
            os.path.join(save_dir, "mlp_ensemble.pt"), map_location=device,
        ))
        model_registry["mlp"] = (
            mlp,
            {k: v.clone() for k, v in mlp.state_dict().items()},
            None,
            {"n_models": cfg.get("ensemble_size", 5), "hidden": cfg.get("mlp_hidden", 256)},
        )

    # Build feature functions
    classical_fn = quantum_fn = None
    if any(m in models_to_eval for m in ("j_mlp", "c_mlp")):
        from physics.openarm_params import compute_openarm_mass_matrix
        from quantum_prior.cached_computer import CachedEntanglementComputer
        qc = CachedEntanglementComputer(
            mass_matrix_fn=compute_openarm_mass_matrix, resolution=0.01,
        )
        classical_fn = qc.get_classical_features
        quantum_fn = qc.get_entanglement_features

    if "j_mlp" in models_to_eval:
        j_mlp = QuantumMLPEnsemble(
            n_models=cfg.get("ensemble_size", 5), hidden=cfg.get("mlp_hidden", 256),
        )
        j_mlp.load_state_dict(torch.load(
            os.path.join(save_dir, "j_mlp_ensemble.pt"), map_location=device,
        ))
        model_registry["j_mlp"] = (
            j_mlp,
            {k: v.clone() for k, v in j_mlp.state_dict().items()},
            classical_fn,
            {"n_models": cfg.get("ensemble_size", 5), "hidden": cfg.get("mlp_hidden", 256)},
        )

    if "c_mlp" in models_to_eval:
        c_mlp = QuantumMLPEnsemble(
            n_models=cfg.get("ensemble_size", 5), hidden=cfg.get("mlp_hidden", 256),
        )
        c_mlp.load_state_dict(torch.load(
            os.path.join(save_dir, "c_mlp_ensemble.pt"), map_location=device,
        ))
        model_registry["c_mlp"] = (
            c_mlp,
            {k: v.clone() for k, v in c_mlp.state_dict().items()},
            quantum_fn,
            {"n_models": cfg.get("ensemble_size", 5), "hidden": cfg.get("mlp_hidden", 256)},
        )

    if "delan" in models_to_eval:
        delan = DeLaN(hidden=cfg.get("delan_hidden", 128))
        delan.load_state_dict(torch.load(
            os.path.join(save_dir, "delan.pt"), map_location=device,
        ))
        model_registry["delan"] = (
            delan,
            {k: v.clone() for k, v in delan.state_dict().items()},
            None,
            {"hidden": cfg.get("delan_hidden", 128)},
        )

    results = {}

    for payload in payloads:
        print(f"\n=== Evaluating payload={payload:.1f} kg ===")
        data_path = os.path.join(data_dir, f"transitions_payload_{payload:.1f}.npz")
        raw = dict(np.load(data_path))

        test_data = {k: v[-n_test:] for k, v in raw.items()}
        ft_data = {k: v[:-n_test] for k, v in raw.items()}

        payload_results = {"payload_kg": payload}

        # CRBA (zero-shot)
        print("  CRBA (zero-shot)...")
        crba = CRBAModel(payload_kg=payload)
        crba_pred = crba.predict_batch(test_data)
        crba_metrics = one_step_rmse(crba_pred, test_data)
        crba_metrics["multi_step_rmse"] = multi_step_rmse_crba(crba, test_data)
        crba_metrics.update(indirect_coupling_rmse(crba_pred, test_data))
        payload_results["crba"] = crba_metrics
        print(f"    1-step RMSE: {crba_metrics['total_rmse']:.6f}")
        print(f"    Indirect RMSE: {crba_metrics['mean_indirect_rmse']:.6f}")

        # Fine-tune sweep for each neural model
        for model_name, (model, base_sd, feat_fn, kwargs) in model_registry.items():
            for n_ft in finetune_sizes:
                key = f"{model_name}_ft_{n_ft}"
                print(f"  {model_name} fine-tune N={n_ft}...")

                metrics = finetune_and_evaluate(
                    model_name, base_sd, ft_data, test_data, n_ft,
                    device=device, feature_fn=feat_fn, **kwargs,
                )
                payload_results[key] = metrics
                print(
                    f"    1-step: {metrics['total_rmse']:.6f}  "
                    f"multi: {metrics['multi_step_rmse']:.6f}  "
                    f"indirect: {metrics['mean_indirect_rmse']:.6f}"
                )

        results[f"payload_{payload:.1f}"] = payload_results

    # Save results
    out_path = os.path.join(save_dir, "transfer_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
