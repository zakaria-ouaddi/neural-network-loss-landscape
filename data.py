"""
Phase I — Step 1.1: Synthetic data generation.

Generates binary classification data via sklearn.datasets.make_classification,
converts to torch tensors, and wraps in a DataLoader.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification

from config import ProjectConfig


def generate_data(config: ProjectConfig):
    """
    Generate synthetic binary classification dataset.

    Parameters
    ----------
    config : ProjectConfig
        Must contain: n_samples, n_features, n_informative, n_classes,
        batch_size, random_seed.

    Returns
    -------
    dataloader : DataLoader
        Batched (X, y) pairs.
    X_full : torch.Tensor, shape (n_samples, n_features), dtype float32
        Full feature matrix (for full-batch loss evaluation).
    y_full : torch.Tensor, shape (n_samples, 1), dtype float32
        Full label vector.
    """
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=config.n_features - config.n_informative,
        n_classes=config.n_classes,
        n_clusters_per_class=1,
        flip_y=0.01,
        random_state=config.random_seed,
    )

    # ── Convert to tensors ────────────────────────────────────────────────
    X_t = torch.from_numpy(X).float()          # (N, n_features)
    y_t = torch.from_numpy(y).float().unsqueeze(1)  # (N, 1)

    # ── Validation ────────────────────────────────────────────────────────
    assert X_t.shape == (config.n_samples, config.n_features), \
        f"X shape mismatch: {X_t.shape}"
    assert y_t.shape == (config.n_samples, 1), \
        f"y shape mismatch: {y_t.shape}"
    unique_labels = torch.unique(y_t)
    assert set(unique_labels.tolist()) <= {0.0, 1.0}, \
        f"Labels must be {{0, 1}}, got {unique_labels.tolist()}"

    dataset = TensorDataset(X_t, y_t)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    return dataloader, X_t, y_t
