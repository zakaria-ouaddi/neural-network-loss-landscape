"""
Phase II — Steps 2.1–2.4: Model definitions and parameter utilities.

Models
------
- LogisticRegressionModel : single nn.Linear, no sigmoid (use BCEWithLogitsLoss)
- MLP                     : Linear → ReLU → Linear, no sigmoid

Utilities
---------
- get_flat_params(model)  : returns 1-D tensor of all parameters
- set_flat_params(model, flat_tensor) : injects a flat vector back into model
"""

import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector


# ═══════════════════════════════════════════════════════════════════════════
# Step 2.1 — Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════

class LogisticRegressionModel(nn.Module):
    """
    Single-layer logistic regression.

    Forward returns raw logits (no sigmoid).
    Pair with nn.BCEWithLogitsLoss.

    Parameters
    ----------
    n_features : int
        Dimensionality of each input sample.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_features)

        Returns
        -------
        logits : Tensor, shape (batch, 1)
        """
        return self.linear(x)


# ═══════════════════════════════════════════════════════════════════════════
# Step 2.2 — Two-layer MLP
# ═══════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Two-layer neural network:  Linear → ReLU → Linear.

    Forward returns raw logits (no sigmoid).
    Non-convex loss landscape due to hidden layer + ReLU.

    Parameters
    ----------
    n_features : int
        Input dimensionality.
    hidden_dim : int
        Number of hidden units.
    """

    def __init__(self, n_features: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_features)

        Returns
        -------
        logits : Tensor, shape (batch, 1)
        """
        return self.fc2(self.relu(self.fc1(x)))


# ═══════════════════════════════════════════════════════════════════════════
# Step 2.3 — Flatten parameters to 1-D vector
# ═══════════════════════════════════════════════════════════════════════════

def get_flat_params(model: nn.Module) -> torch.Tensor:
    """
    Concatenate all model parameters into a single 1-D tensor.

    Returns
    -------
    flat : Tensor, shape (D,)
        D = total number of scalar parameters.
    """
    flat = parameters_to_vector(model.parameters())
    # Validation
    expected_d = sum(p.numel() for p in model.parameters())
    assert flat.shape == (expected_d,), \
        f"Flat param shape {flat.shape} != expected ({expected_d},)"
    return flat


# ═══════════════════════════════════════════════════════════════════════════
# Step 2.4 — Inject flat vector back into model
# ═══════════════════════════════════════════════════════════════════════════

def set_flat_params(model: nn.Module, flat_tensor: torch.Tensor) -> None:
    """
    Write a 1-D parameter vector into *model* in-place.

    Parameters
    ----------
    model : nn.Module
    flat_tensor : Tensor, shape (D,)
        Must match the total parameter count of *model*.
    """
    # Ensure correct device and dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    flat_tensor = flat_tensor.to(device=device, dtype=dtype)
    vector_to_parameters(flat_tensor, model.parameters())
