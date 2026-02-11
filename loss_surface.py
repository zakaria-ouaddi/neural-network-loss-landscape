"""
Phase VI — Steps 6.1–6.2: Loss surface computation.

Receives **already filter-normalised** u, v from projection.py.

For each grid point (α, β):
    θ_new  =  θ*  +  α·u  +  β·v
    Z[i,j] =  L(θ_new; X, y)       (full-batch, no mini-batch noise)

Original model parameters are saved and restored after the sweep.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models import set_flat_params, get_flat_params


def compute_loss_surface(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    final_theta: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    vis_resolution: int = 50,
    vis_range: float = 1.0,
) -> tuple:
    """
    Compute loss over a 2-D grid in the PCA plane.

    Parameters
    ----------
    model : nn.Module
        Architecture reference (weights will be temporarily overwritten).
    X : Tensor, shape (N, n_features)
        Full training features.
    y : Tensor, shape (N, 1)
        Full training labels.
    final_theta : Tensor, shape (D,)
        θ* — centre of the grid.
    u, v : Tensor, shape (D,)
        Filter-normalised PCA basis vectors.
    vis_resolution : int
        Number of grid points per axis.
    vis_range : float
        Grid spans [−vis_range, +vis_range] along each axis.

    Returns
    -------
    alpha_grid : ndarray, shape (res, res)
    beta_grid  : ndarray, shape (res, res)
    Z          : ndarray, shape (res, res)
        Loss values.
    """
    # ── Step 6.1 — Create 2-D meshgrid ────────────────────────────────────
    alphas = np.linspace(-vis_range, vis_range, vis_resolution)
    betas  = np.linspace(-vis_range, vis_range, vis_resolution)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)

    Z = np.zeros_like(alpha_grid)

    criterion = nn.BCEWithLogitsLoss()

    # ── Save original parameters ──────────────────────────────────────────
    original_params = get_flat_params(model).detach().cpu().clone()

    # ── Step 6.2 — Sweep grid ────────────────────────────────────────────
    model.eval()
    print("\nComputing loss surface...")
    for i in tqdm(range(vis_resolution), desc="Loss surface rows", leave=False):
        for j in range(vis_resolution):
            alpha = alpha_grid[i, j]
            beta  = beta_grid[i, j]

            # θ_new = θ* + α·u + β·v
            theta_new = final_theta + alpha * u + beta * v
            set_flat_params(model, theta_new)

            with torch.no_grad():
                logits = model(X)
                loss = criterion(logits, y)
                Z[i, j] = loss.item()

    # ── Restore original parameters ──────────────────────────────────────
    set_flat_params(model, original_params)

    return alpha_grid, beta_grid, Z
