"""MLP Ensemble world model (PETS-style).

5-member ensemble, each MLP predicts (delta_q, delta_dq) given (q, dq, tau).
Disagreement = epistemic uncertainty.

Variants:
- MLPEnsemble(input_dim=21): base MLP (q, dq, tau)
- QuantumMLPEnsemble(input_dim=42): augmented with 21-dim structure features
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """Single MLP: (q, dq, tau) -> (delta_q, delta_dq)."""

    def __init__(self, input_dim: int = 21, output_dim: int = 14, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPEnsemble(nn.Module):
    """Ensemble of K MLP models for epistemic uncertainty."""

    def __init__(self, n_models: int = 5, input_dim: int = 21, output_dim: int = 14, hidden: int = 256):
        super().__init__()
        self.models = nn.ModuleList([
            MLPModel(input_dim, output_dim, hidden) for _ in range(n_models)
        ])
        self.n_models = n_models

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns mean prediction across ensemble. Shape: (batch, output_dim)."""
        preds = torch.stack([m(x) for m in self.models], dim=0)  # (K, batch, out)
        return preds.mean(dim=0)

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) predictions."""
        preds = torch.stack([m(x) for m in self.models], dim=0)
        return preds.mean(dim=0), preds.std(dim=0)

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Predict state delta: x contains [q, dq, tau], returns [dq, ddq]."""
        return self.forward(x)


class QuantumMLPEnsemble(MLPEnsemble):
    """MLP Ensemble with quantum structure features.

    Input: [q(7), dq(7), tau(7), features(21)] = 42-dim
    Output: [q_next(7), dq_next(7)] = 14-dim

    Same architecture as MLPEnsemble, just wider input.
    """

    def __init__(
        self,
        n_models: int = 5,
        output_dim: int = 14,
        hidden: int = 256,
    ):
        super().__init__(
            n_models=n_models,
            input_dim=42,
            output_dim=output_dim,
            hidden=hidden,
        )


def train_ensemble(
    model: MLPEnsemble,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
) -> list[float]:
    """Train ensemble with bootstrap sampling.

    Each member sees a different bootstrap resample of each batch.
    Returns list of per-epoch average losses.
    """
    model = model.to(device)
    optimizers = [
        torch.optim.Adam(m.parameters(), lr=lr) for m in model.models
    ]

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.shape[0]

            for k, (member, opt) in enumerate(zip(model.models, optimizers)):
                # Bootstrap: resample with replacement
                idx = torch.randint(0, batch_size, (batch_size,), device=device)
                x_k = x_batch[idx]
                y_k = y_batch[idx]

                pred = member(x_k)
                loss = nn.functional.mse_loss(pred, y_k)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

            n_batches += 1

        avg = total_loss / (n_batches * model.n_models)
        epoch_losses.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.6f}")

    return epoch_losses
