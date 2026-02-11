"""
Phase XI (merged into Phase V, Step 5.3b): Per-layer filter normalization.

For each layer l, given a direction vector d and the reference parameter θ*:

    d_l  ←  d_l  ·  (‖θ*_l‖ / ‖d_l‖)

This removes the scale ambiguity between layers so that the visualised
loss surface faithfully represents the geometry near θ*.

Reference: Li et al., "Visualizing the Loss Landscape of Neural Nets", 2018.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def _get_layer_slices(model: nn.Module) -> List[Tuple[int, int, torch.Size]]:
    """
    Compute (start, end, shape) for every parameter tensor in *model*.

    Returns
    -------
    slices : list[(start, end, shape)]
    """
    slices = []
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        slices.append((offset, offset + numel, p.shape))
        offset += numel
    return slices


def filter_normalize(
    model: nn.Module,
    direction: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Apply per-layer filter normalization to a single direction vector.

    Parameters
    ----------
    model : nn.Module
        Used only to determine per-layer boundaries and shapes.
    direction : Tensor, shape (D,)
        Raw direction vector (e.g. a PCA component).
    reference : Tensor, shape (D,)
        Reference parameter vector (typically θ*, the final trained weights).

    Returns
    -------
    normed : Tensor, shape (D,)
        Direction with each layer segment rescaled so that
        ‖d_l‖ is proportional to ‖θ*_l‖.
    """
    slices = _get_layer_slices(model)
    normed = direction.clone()

    for start, end, _ in slices:
        d_l = direction[start:end].float()
        theta_l = reference[start:end].float()

        d_norm = torch.norm(d_l)
        theta_norm = torch.norm(theta_l)

        # Avoid division by zero for uninitialised / zero layers
        if d_norm > 1e-10:
            normed[start:end] = d_l * (theta_norm / d_norm)

    return normed


def filter_normalize_uv(
    model: nn.Module,
    u: torch.Tensor,
    v: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper: normalise both PCA directions at once.

    Parameters
    ----------
    model : nn.Module
    u, v : Tensor, shape (D,)
        Raw PCA basis vectors.
    reference : Tensor, shape (D,)
        θ* (final trained weights).

    Returns
    -------
    u_normed, v_normed : Tensor, shape (D,)
    """
    return filter_normalize(model, u, reference), \
           filter_normalize(model, v, reference)
