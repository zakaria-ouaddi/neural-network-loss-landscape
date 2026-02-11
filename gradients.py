"""
Phase VII — Step 7.1: Gradient projection onto PCA plane.

At each training checkpoint θ_t:
1. Compute the full-batch gradient g_t.
2. Project g_t onto the PCA directions u, v.
3. Store the *negative* projection (−g_u, −g_v) because gradient descent
   moves *against* the gradient.

Output is used by visualization.py to draw quiver arrows.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from models import set_flat_params, get_flat_params


def compute_projected_gradients(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    trajectory: List[torch.Tensor],
    u: torch.Tensor,
    v: torch.Tensor,
    sample_every: int = 1,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Compute projected gradient arrows for each sampled checkpoint.

    Parameters
    ----------
    model : nn.Module
    X : Tensor, shape (N, n_features)
    y : Tensor, shape (N, 1)
    trajectory : list[Tensor], each shape (D,)
    u, v : Tensor, shape (D,)
        Filter-normalised PCA directions.
    sample_every : int
        Compute gradients every N-th checkpoint (to avoid clutter).

    Returns
    -------
    neg_gu_list : list[float]
        −⟨g_t, u⟩ for each sampled step.
    neg_gv_list : list[float]
        −⟨g_t, v⟩ for each sampled step.
    indices : list[int]
        Which trajectory indices were sampled.
    """
    criterion = nn.BCEWithLogitsLoss()
    original_params = get_flat_params(model).detach().cpu().clone()

    neg_gu_list: List[float] = []
    neg_gv_list: List[float] = []
    indices: List[int] = []

    for t in range(0, len(trajectory) - 1, sample_every):
        # Set model to θ_t
        set_flat_params(model, trajectory[t])

        model.train()
        model.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        # Flatten gradient
        grad_flat = torch.cat([
            p.grad.detach().cpu().flatten()
            for p in model.parameters()
            if p.grad is not None
        ])

        # Project
        g_u = torch.dot(grad_flat, u).item()
        g_v = torch.dot(grad_flat, v).item()

        neg_gu_list.append(-g_u)
        neg_gv_list.append(-g_v)
        indices.append(t)

    # Restore
    set_flat_params(model, original_params)

    return neg_gu_list, neg_gv_list, indices
