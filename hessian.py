"""
Phase X — Steps 10.1–10.2: Hessian analysis.

10.1  Hessian-vector product  Hv = ∇_θ(∇_θ L · v)
10.2  Power iteration to estimate the largest eigenvalue λ_max
      (optionally tracked across training checkpoints)
"""

import torch
import torch.nn as nn
from typing import List

from models import set_flat_params, get_flat_params


# ═══════════════════════════════════════════════════════════════════════════
# Step 10.1 — Hessian-vector product
# ═══════════════════════════════════════════════════════════════════════════

def hessian_vector_product(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Hessian-vector product Hv without materialising H.

    H v  =  ∇_θ ( ∇_θ L(θ)  ·  v )

    Parameters
    ----------
    model : nn.Module
        Parameters must be at the point where Hv is desired.
    X : Tensor, shape (N, n_features)
    y : Tensor, shape (N, 1)
    vec : Tensor, shape (D,)
        The direction vector v.

    Returns
    -------
    hv : Tensor, shape (D,)
    """
    criterion = nn.BCEWithLogitsLoss()

    # First backward: get ∇_θ L
    model.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])   # (D,)

    # Scalar product g · v
    gv = torch.dot(flat_grad, vec)

    # Second backward: ∇_θ (g · v)  =  Hv
    hv_parts = torch.autograd.grad(gv, model.parameters())
    hv = torch.cat([h.reshape(-1) for h in hv_parts]).detach()

    return hv


# ═══════════════════════════════════════════════════════════════════════════
# Step 10.2 — Power iteration for λ_max
# ═══════════════════════════════════════════════════════════════════════════

def power_iteration(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    num_iters: int = 20,
) -> float:
    """
    Estimate the largest eigenvalue of the Hessian via power iteration.

    Parameters
    ----------
    model : nn.Module
    X, y : Tensor
    num_iters : int
        Number of power iteration steps.

    Returns
    -------
    eigenvalue : float
        Approximate largest |eigenvalue| of the Hessian at current θ.
    """
    D = sum(p.numel() for p in model.parameters())
    # Random initial vector
    v = torch.randn(D)
    v = v / torch.norm(v)

    eigenvalue = 0.0
    for _ in range(num_iters):
        hv = hessian_vector_product(model, X, y, v)
        eigenvalue = torch.dot(v, hv).item()
        v = hv / (torch.norm(hv) + 1e-12)

    return eigenvalue


def compute_eigenvalues_along_trajectory(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    trajectory: List[torch.Tensor],
    sample_every: int = 5,
    power_iters: int = 20,
) -> List[float]:
    """
    Estimate λ_max at sampled checkpoints along the trajectory.

    Parameters
    ----------
    model : nn.Module
    X, y : Tensor
    trajectory : list[Tensor]
    sample_every : int
    power_iters : int

    Returns
    -------
    eigenvalues : list[float]
        One value per trajectory step (un-sampled steps get linearly
        interpolated for plotting convenience).
    """
    original = get_flat_params(model).detach().cpu().clone()
    sampled_eigs = {}

    for t in range(0, len(trajectory), sample_every):
        set_flat_params(model, trajectory[t])
        eig = power_iteration(model, X, y, num_iters=power_iters)
        sampled_eigs[t] = eig

    # Restore
    set_flat_params(model, original)

    # Linearly interpolate for all steps
    eigenvalues = []
    sorted_keys = sorted(sampled_eigs.keys())
    for t in range(len(trajectory)):
        if t in sampled_eigs:
            eigenvalues.append(sampled_eigs[t])
        else:
            # Find surrounding sampled points
            lo = max(k for k in sorted_keys if k <= t)
            hi = min((k for k in sorted_keys if k > t), default=lo)
            if lo == hi:
                eigenvalues.append(sampled_eigs[lo])
            else:
                frac = (t - lo) / (hi - lo)
                eigenvalues.append(
                    sampled_eigs[lo] * (1 - frac) + sampled_eigs[hi] * frac
                )

    return eigenvalues
