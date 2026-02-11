"""
Phase 0 — Step 0.2: Project configuration dataclass.

Single source of truth for every tuneable parameter in the project.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ProjectConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    n_samples: int = 1000
    n_features: int = 10
    n_informative: int = 8
    n_classes: int = 2
    batch_size: int = 256
    random_seed: int = 42

    # ── Architecture ──────────────────────────────────────────────────────
    model_type: str = "logistic"        # 'logistic' | 'mlp'
    hidden_dim: int = 32

    # ── Optimisation ──────────────────────────────────────────────────────
    lr: float = 0.05
    epochs: int = 100
    optimizer: str = "SGD"              # 'SGD' | 'SGD_momentum' | 'Adam'

    # ── Visualisation ─────────────────────────────────────────────────────
    vis_resolution: int = 50            # grid points per axis for loss surface
    vis_range: float = 1.0              # ±range in PCA space
    projection_method: str = "pca"      # 'pca' | 'isomap'

    # ── Multi-seed experiment ─────────────────────────────────────────────
    seed_list: List[int] = field(default_factory=lambda: list(range(10)))
