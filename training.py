"""
Phase III — Step 3.1: Training loop with trajectory collection.
Phase IV  — Step 4.1: Multi-seed experiment wrapper.

Key contract
------------
- trajectory[0]  = initial (random) parameters
- trajectory[-1] = parameters after the last optimizer.step()
- len(trajectory) == epochs + 1
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from config import ProjectConfig
from models import (
    LogisticRegressionModel,
    MLP,
    get_flat_params,
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 3.1 — Single-run training with trajectory collection
# ═══════════════════════════════════════════════════════════════════════════

def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: ProjectConfig,
) -> List[torch.Tensor]:
    """
    Train *model* and return the full parameter trajectory.

    Parameters
    ----------
    model : nn.Module
        Model to train (already on correct device).
    dataloader : DataLoader
        Batched (X, y) pairs.
    config : ProjectConfig
        Must contain: lr, epochs, optimizer.

    Returns
    -------
    trajectory : list[Tensor]
        Each element is a 1-D detached CPU tensor of shape (D,).
        Length = config.epochs + 1 (initial + one per epoch).
    """
    # ── Optimizer ─────────────────────────────────────────────────────────
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    elif config.optimizer == "SGD_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    criterion = nn.BCEWithLogitsLoss()

    # ── Trajectory storage ────────────────────────────────────────────────
    trajectory: List[torch.Tensor] = []

    # Append initial parameters (before any training)
    trajectory.append(get_flat_params(model).detach().cpu().clone())

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in tqdm(range(config.epochs), desc="Training", leave=False):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Append parameters after this epoch's updates
        trajectory.append(get_flat_params(model).detach().cpu().clone())

    # ── Validation ────────────────────────────────────────────────────────
    assert len(trajectory) == config.epochs + 1, (
        f"Trajectory length {len(trajectory)} != expected {config.epochs + 1}"
    )

    return trajectory


# ═══════════════════════════════════════════════════════════════════════════
# Step 4.1 — Multi-seed experiment
# ═══════════════════════════════════════════════════════════════════════════

def _make_model(config: ProjectConfig) -> nn.Module:
    """Instantiate the model specified by *config.model_type*."""
    if config.model_type == "logistic":
        return LogisticRegressionModel(config.n_features)
    elif config.model_type == "mlp":
        return MLP(config.n_features, config.hidden_dim)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


def train_multiple_seeds(
    config: ProjectConfig,
    dataloader: torch.utils.data.DataLoader,
) -> Dict[int, List[torch.Tensor]]:
    """
    Train from scratch for every seed in *config.seed_list*.

    Parameters
    ----------
    config : ProjectConfig
    dataloader : DataLoader

    Returns
    -------
    all_trajectories : dict[int, list[Tensor]]
        Mapping seed → trajectory.  Each trajectory has length epochs + 1.
    """
    all_trajectories: Dict[int, List[torch.Tensor]] = {}

    for seed in config.seed_list:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        # Deterministic init
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = _make_model(config)
        trajectory = train(model, dataloader, config)

        # Confirm independent initialisation by checking first params differ
        all_trajectories[seed] = trajectory

    # Quick sanity: all initial params should differ across seeds
    init_params = [t[0] for t in all_trajectories.values()]
    for i in range(1, len(init_params)):
        assert not torch.equal(init_params[0], init_params[i]), \
            f"Seed {config.seed_list[i]} produced same init as seed {config.seed_list[0]}"

    return all_trajectories
