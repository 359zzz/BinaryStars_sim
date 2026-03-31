"""PyTorch dataset for dynamics transitions."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """Dataset of (q, dq, tau) -> (q_next, dq_next) transitions.

    Input:  [q(7), dq(7), tau(7)] = 21-dim
    Target: [q_next(7), dq_next(7)] = 14-dim
    """

    def __init__(self, data: dict[str, np.ndarray], max_samples: int | None = None):
        n = data["q"].shape[0]
        if max_samples is not None and max_samples < n:
            idx = np.random.permutation(n)[:max_samples]
            data = {k: v[idx] for k, v in data.items()}

        self.x = torch.from_numpy(
            np.concatenate([data["q"], data["dq"], data["tau"]], axis=1)
        ).float()
        self.y = torch.from_numpy(
            np.concatenate([data["q_next"], data["dq_next"]], axis=1)
        ).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class QuantumTransitionDataset(Dataset):
    """Transition dataset augmented with quantum structure features.

    Input:  [q(7), dq(7), tau(7), features(21)] = 42-dim
    Target: [q_next(7), dq_next(7)] = 14-dim

    Features can be either |J_ij| (classical) or C_ij (quantum).
    Precomputed offline for efficiency.

    Parameters
    ----------
    data : dict with keys q, dq, tau, q_next, dq_next
    feature_fn : callable q -> (21,) feature vector
    max_samples : optional cap on dataset size
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        feature_fn,
        max_samples: int | None = None,
    ):
        n = data["q"].shape[0]
        if max_samples is not None and max_samples < n:
            idx = np.random.permutation(n)[:max_samples]
            data = {k: v[idx] for k, v in data.items()}
            n = max_samples

        # Precompute structure features for all q values
        features = np.zeros((n, 21), dtype=np.float32)
        for i in range(n):
            features[i] = feature_fn(data["q"][i])

        self.x = torch.from_numpy(
            np.concatenate(
                [data["q"], data["dq"], data["tau"], features], axis=1
            )
        ).float()
        self.y = torch.from_numpy(
            np.concatenate([data["q_next"], data["dq_next"]], axis=1)
        ).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_npz(path: str, max_samples: int | None = None) -> TransitionDataset:
    """Load transition dataset from .npz file."""
    data = dict(np.load(path))
    return TransitionDataset(data, max_samples=max_samples)
