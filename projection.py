"""
Phase V  — Steps 5.1–5.4: PCA projection (with filter normalization).
Phase IX — Steps 9.1–9.2: Isomap projection.

Workflow
--------
1. Stack trajectory tensor  (T, D)
2. Center around θ* (final state)
3. Fit PCA, extract u, v
4. **Filter-normalise u, v** (Step 5.3b — imported from filter_norm.py)
5. Project trajectory → 2D (α, β)
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from typing import Tuple, Optional
import torch.nn as nn

from filter_norm import filter_normalize_uv


# ═══════════════════════════════════════════════════════════════════════════
# Step 5.1–5.4 — PCA projection pipeline
# ═══════════════════════════════════════════════════════════════════════════

def pca_project(
    trajectory: list,
    model: nn.Module,
    n_components: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    PCA-project the trajectory onto a 2-D plane centred at θ*.

    Parameters
    ----------
    trajectory : list[Tensor]
        Length T+1 list of flat parameter vectors (shape D each).
    model : nn.Module
        Needed for per-layer filter normalization only.
    n_components : int
        Number of PCA components (default 2).

    Returns
    -------
    coords : Tensor, shape (T+1, 2)
        (α, β) coordinates of each checkpoint.
    u : Tensor, shape (D,)
        First (filter-normalised) PCA direction.
    v : Tensor, shape (D,)
        Second (filter-normalised) PCA direction.
    final_theta : Tensor, shape (D,)
        θ* (final trained weights, used as origin).
    explained_variance : ndarray, shape (n_components,)
        Fraction of variance explained by each component.
    """
    # Step 5.1 — Stack
    trajectory_tensor = torch.stack(trajectory)          # (T+1, D)

    # Step 5.2 — Centre around final state
    final_theta = trajectory_tensor[-1].clone()          # (D,)
    centred = trajectory_tensor - final_theta             # (T+1, D)

    # Step 5.3 — PCA
    pca = PCA(n_components=n_components)
    pca.fit(centred.numpy())

    components = torch.from_numpy(pca.components_).float()  # (2, D)
    u_raw = components[0]                                    # (D,)
    v_raw = components[1]                                    # (D,)

    assert u_raw.shape == (trajectory_tensor.shape[1],)
    assert v_raw.shape == (trajectory_tensor.shape[1],)

    # Step 5.3b — Filter normalization (Phase XI, moved here)
    u, v = filter_normalize_uv(model, u_raw, v_raw, final_theta)

    # Step 5.4 — Project trajectory to 2D
    coords = torch.zeros(len(trajectory), 2)
    for t in range(len(trajectory)):
        diff = trajectory_tensor[t] - final_theta
        coords[t, 0] = torch.dot(diff, u)
        coords[t, 1] = torch.dot(diff, v)

    explained_variance = pca.explained_variance_ratio_

    print(f"\n{'─'*50}")
    print(f"  PCA Projection Summary")
    print(f"{'─'*50}")
    print(f"  Original dimensionality : {trajectory_tensor.shape[1]}")
    print(f"  PC1 variance explained  : {explained_variance[0]:.4f}")
    print(f"  PC2 variance explained  : {explained_variance[1]:.4f}")
    print(f"  Total variance explained: {explained_variance.sum():.4f}")
    print(f"  Filter normalization    : applied")
    print(f"{'─'*50}")

    return coords, u, v, final_theta, explained_variance


# ═══════════════════════════════════════════════════════════════════════════
# Phase IX — Isomap projection
# ═══════════════════════════════════════════════════════════════════════════

def isomap_project(
    trajectory: list,
    n_components: int = 2,
    n_neighbors: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    Project trajectory using Isomap (non-linear manifold method).

    Parameters
    ----------
    trajectory : list[Tensor]
    n_components : int
    n_neighbors : int

    Returns
    -------
    coords_iso : ndarray, shape (T+1, 2)
    residual_variance : float
        Reconstruction error of the Isomap embedding.
    """
    trajectory_tensor = torch.stack(trajectory).numpy()   # (T+1, D)

    # Clamp n_neighbors to valid range
    max_nn = max(1, len(trajectory) - 1)
    n_neighbors = min(n_neighbors, max_nn)

    iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    coords_iso = iso.fit_transform(trajectory_tensor)

    residual_variance = 1.0 - iso.reconstruction_error() if hasattr(iso, 'reconstruction_error') else float('nan')
    # sklearn Isomap stores reconstruction_error_ attribute
    if hasattr(iso, 'reconstruction_error_'):
        residual_variance = iso.reconstruction_error_

    print(f"\n{'─'*50}")
    print(f"  Isomap Projection Summary")
    print(f"{'─'*50}")
    print(f"  n_neighbors             : {n_neighbors}")
    print(f"  Residual variance       : {residual_variance:.6f}")
    print(f"{'─'*50}")

    return coords_iso, residual_variance
