#!/usr/bin/env python3
"""
main.py — End-to-end orchestrator for Gradient Descent Visualisation.

Executes every phase from the blueprint in order:
  Phase 0   Global setup
  Phase I   Data generation
  Phase II  (models — imported)
  Phase III Training + trajectory
  Phase IV  Multi-seed experiment
  Phase V   PCA projection (with filter normalisation)
  Phase VI  Loss surface
  Phase VII Gradient projection
  Phase VIII 3-D visualisation
  Phase IX  Isomap comparison
  Phase X   Hessian analysis
  Phase XII Optional extensions (optimizer comparison)
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from config import ProjectConfig
from data import generate_data
from models import LogisticRegressionModel, MLP, get_flat_params, set_flat_params
from training import train, train_multiple_seeds, _make_model
from projection import pca_project, isomap_project
from loss_surface import compute_loss_surface
from gradients import compute_projected_gradients
from hessian import compute_eigenvalues_along_trajectory
from visualization import (
    plot_loss_surface_3d,
    plot_multiple_seeds,
    plot_pca_vs_isomap,
    plot_curvature_trajectory,
)
from animations import create_gd_animation, create_multi_seed_animation
from interactive_plots import create_interactive_surface, create_interactive_multi_seed

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _set_seed(seed: int) -> None:
    """Phase 0 — deterministic behaviour."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════
#  A. Logistic Regression — Convex (Tasks 1-3)
# ══════════════════════════════════════════════════════════════════════════

def run_logistic(config: ProjectConfig) -> None:
    print("\n" + "=" * 70)
    print("  A. LOGISTIC REGRESSION (convex)")
    print("=" * 70)

    _set_seed(config.random_seed)

    # Phase I — data
    dl, X, y = generate_data(config)

    # Phase III — training
    model = LogisticRegressionModel(config.n_features)
    trajectory = train(model, dl, config)

    # Phase V — PCA projection (filter-normalised)
    coords, u, v, final_theta, pca_var = pca_project(trajectory, model)

    # Phase VI — loss surface
    alpha_grid, beta_grid, Z = compute_loss_surface(
        model, X, y, final_theta, u, v,
        vis_resolution=config.vis_resolution,
        vis_range=config.vis_range,
    )

    # Phase VII — gradient projection
    sample_every = max(1, config.epochs // 15)
    neg_gu, neg_gv, grad_idx = compute_projected_gradients(
        model, X, y, trajectory, u, v, sample_every=sample_every,
    )

    # Phase VIII — 3D visualisation
    plot_loss_surface_3d(
        alpha_grid, beta_grid, Z, coords,
        neg_gu=neg_gu, neg_gv=neg_gv, grad_indices=grad_idx,
        title="Loss Surface — Logistic Regression (Convex)",
        save_path=os.path.join(FIGURES_DIR, "logistic_3d.png"),
    )

    # Animated GIF
    create_gd_animation(
        alpha_grid, beta_grid, Z, coords,
        neg_gu=neg_gu, neg_gv=neg_gv, grad_indices=grad_idx,
        title="GD Animation — Logistic Regression",
        save_path=os.path.join(FIGURES_DIR, "logistic_animation.gif"),
    )

    # Interactive Plotly HTML
    create_interactive_surface(
        alpha_grid, beta_grid, Z, coords,
        neg_gu=neg_gu, neg_gv=neg_gv, grad_indices=grad_idx,
        title="Interactive Loss Surface — Logistic Regression",
        save_path=os.path.join(FIGURES_DIR, "logistic_interactive.html"),
    )

    # Phase IX — Isomap comparison
    iso_coords, iso_resid = isomap_project(trajectory)
    plot_pca_vs_isomap(
        coords, iso_coords, pca_var, iso_resid,
        title="Logistic Regression — PCA vs Isomap",
        save_path=os.path.join(FIGURES_DIR, "logistic_pca_vs_isomap.png"),
    )

    print("  ✓ Logistic regression complete")


# ══════════════════════════════════════════════════════════════════════════
#  B. Two-layer MLP — Non-convex (Task 4)
# ══════════════════════════════════════════════════════════════════════════

def run_mlp_single(config: ProjectConfig) -> None:
    print("\n" + "=" * 70)
    print("  B. MLP — SINGLE SEED (non-convex)")
    print("=" * 70)

    cfg = ProjectConfig(
        **{**config.__dict__, "model_type": "mlp"},
    )
    _set_seed(cfg.random_seed)

    dl, X, y = generate_data(cfg)
    model = MLP(cfg.n_features, cfg.hidden_dim)
    trajectory = train(model, dl, cfg)

    coords, u, v, final_theta, pca_var = pca_project(trajectory, model)

    alpha_grid, beta_grid, Z = compute_loss_surface(
        model, X, y, final_theta, u, v,
        vis_resolution=cfg.vis_resolution,
        vis_range=cfg.vis_range,
    )

    sample_every = max(1, cfg.epochs // 15)
    neg_gu, neg_gv, grad_idx = compute_projected_gradients(
        model, X, y, trajectory, u, v, sample_every=sample_every,
    )

    plot_loss_surface_3d(
        alpha_grid, beta_grid, Z, coords,
        neg_gu=neg_gu, neg_gv=neg_gv, grad_indices=grad_idx,
        title="Loss Surface — MLP (Non-Convex)",
        save_path=os.path.join(FIGURES_DIR, "mlp_3d.png"),
    )

    # Animated GIF
    create_gd_animation(
        alpha_grid, beta_grid, Z, coords,
        neg_gu=neg_gu, neg_gv=neg_gv, grad_indices=grad_idx,
        title="GD Animation — MLP (Non-Convex)",
        save_path=os.path.join(FIGURES_DIR, "mlp_animation.gif"),
    )

    # Interactive Plotly HTML
    create_interactive_surface(
        alpha_grid, beta_grid, Z, coords,
        neg_gu=neg_gu, neg_gv=neg_gv, grad_indices=grad_idx,
        title="Interactive Loss Surface — MLP",
        save_path=os.path.join(FIGURES_DIR, "mlp_interactive.html"),
    )

    # Phase IX — Isomap comparison
    iso_coords, iso_resid = isomap_project(trajectory)
    plot_pca_vs_isomap(
        coords, iso_coords, pca_var, iso_resid,
        title="MLP — PCA vs Isomap",
        save_path=os.path.join(FIGURES_DIR, "mlp_pca_vs_isomap.png"),
    )

    # Phase X — Hessian curvature
    print("\n  Computing Hessian eigenvalues along trajectory...")
    eigenvalues = compute_eigenvalues_along_trajectory(
        model, X, y, trajectory, sample_every=max(1, cfg.epochs // 10),
    )
    plot_curvature_trajectory(
        coords, eigenvalues,
        title="MLP — Curvature-Coloured Trajectory",
        save_path=os.path.join(FIGURES_DIR, "mlp_curvature.png"),
    )

    print("  ✓ MLP single-seed complete")


# ══════════════════════════════════════════════════════════════════════════
#  C. Multi-seed MLP (Task 5)
# ══════════════════════════════════════════════════════════════════════════

def run_mlp_multi_seed(config: ProjectConfig) -> None:
    print("\n" + "=" * 70)
    print("  C. MLP — MULTI-SEED")
    print("=" * 70)

    cfg = ProjectConfig(
        **{**config.__dict__, "model_type": "mlp"},
    )

    # Phase I — data (single dataset for all seeds)
    _set_seed(cfg.random_seed)
    dl, X, y = generate_data(cfg)

    # Phase IV — multi-seed training
    all_trajectories = train_multiple_seeds(cfg, dl)

    # Combine all weight checkpoints for a shared PCA basis
    all_points = []
    for traj in all_trajectories.values():
        all_points.extend(traj)

    # Use the first seed's final model for architecture reference
    first_seed = cfg.seed_list[0]
    torch.manual_seed(first_seed)
    ref_model = MLP(cfg.n_features, cfg.hidden_dim)
    set_flat_params(ref_model, all_trajectories[first_seed][-1])

    # Phase V — shared PCA
    coords_ref, u, v, final_theta, pca_var = pca_project(
        all_trajectories[first_seed], ref_model,
    )

    # Project every trajectory into the same PCA plane
    all_coords = {}
    for seed, traj in all_trajectories.items():
        traj_tensor = torch.stack(traj)
        c = torch.zeros(len(traj), 2)
        for t in range(len(traj)):
            diff = traj_tensor[t] - final_theta
            c[t, 0] = torch.dot(diff, u)
            c[t, 1] = torch.dot(diff, v)
        all_coords[seed] = c

    # Phase VI — loss surface (shared basis)
    alpha_grid, beta_grid, Z = compute_loss_surface(
        ref_model, X, y, final_theta, u, v,
        vis_resolution=cfg.vis_resolution,
        vis_range=cfg.vis_range,
    )

    # Phase VIII — multi-seed overlay
    plot_multiple_seeds(
        alpha_grid, beta_grid, Z, all_coords,
        title="MLP — Multiple Seeds (Non-Convex)",
        save_path=os.path.join(FIGURES_DIR, "mlp_multi_seed.png"),
    )

    # Animated GIF — multi-seed
    create_multi_seed_animation(
        alpha_grid, beta_grid, Z, all_coords,
        title="Multi-Seed GD Animation",
        save_path=os.path.join(FIGURES_DIR, "mlp_multi_seed_animation.gif"),
    )

    # Interactive Plotly — multi-seed
    create_interactive_multi_seed(
        alpha_grid, beta_grid, Z, all_coords,
        title="Interactive Multi-Seed — MLP",
        save_path=os.path.join(FIGURES_DIR, "mlp_multi_seed_interactive.html"),
    )

    print("  ✓ MLP multi-seed complete")


# ══════════════════════════════════════════════════════════════════════════
#  D. Phase XII — Optional: Optimizer Comparison
# ══════════════════════════════════════════════════════════════════════════

def run_optimizer_comparison(config: ProjectConfig) -> None:
    print("\n" + "=" * 70)
    print("  D. OPTIMIZER COMPARISON (optional)")
    print("=" * 70)

    _set_seed(config.random_seed)
    dl, X, y = generate_data(config)

    optimizers = ["SGD", "SGD_momentum", "Adam"]
    trajectories = {}

    for opt_name in optimizers:
        print(f"\n  Training with {opt_name}...")
        _set_seed(config.random_seed)  # same init for fair comparison
        cfg_opt = ProjectConfig(**{**config.__dict__, "model_type": "mlp", "optimizer": opt_name})
        model = MLP(cfg_opt.n_features, cfg_opt.hidden_dim)
        traj = train(model, dl, cfg_opt)
        trajectories[opt_name] = traj

    # Shared PCA from SGD trajectory
    ref_model = MLP(config.n_features, config.hidden_dim)
    set_flat_params(ref_model, trajectories["SGD"][-1])

    all_points = []
    for traj in trajectories.values():
        all_points.extend(traj)

    coords_ref, u, v, final_theta, _ = pca_project(all_points, ref_model)

    # Project each optimizer's trajectory
    all_coords = {}
    for name, traj in trajectories.items():
        traj_t = torch.stack(traj)
        c = torch.zeros(len(traj), 2)
        for t in range(len(traj)):
            diff = traj_t[t] - final_theta
            c[t, 0] = torch.dot(diff, u)
            c[t, 1] = torch.dot(diff, v)
        all_coords[name] = c

    # Loss surface
    alpha_grid, beta_grid, Z = compute_loss_surface(
        ref_model, X, y, final_theta, u, v,
        vis_resolution=config.vis_resolution,
        vis_range=config.vis_range,
    )

    # Reuse multi-seed plotter (seed label → optimizer name)
    plot_multiple_seeds(
        alpha_grid, beta_grid, Z,
        {name: coords for name, coords in all_coords.items()},
        title="MLP — Optimizer Comparison",
        save_path=os.path.join(FIGURES_DIR, "optimizer_comparison.png"),
    )

    # Interactive Plotly — optimizer comparison
    create_interactive_multi_seed(
        alpha_grid, beta_grid, Z,
        {name: coords for name, coords in all_coords.items()},
        title="Interactive Optimizer Comparison",
        save_path=os.path.join(FIGURES_DIR, "optimizer_comparison_interactive.html"),
    )

    print("  ✓ Optimizer comparison complete")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    config = ProjectConfig()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   GRADIENT DESCENT LOSS LANDSCAPE VISUALISATION            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Config: {config}\n")

    run_logistic(config)
    run_mlp_single(config)
    run_mlp_multi_seed(config)
    run_optimizer_comparison(config)

    print("\n" + "=" * 70)
    print(f"  ALL PHASES COMPLETE — figures saved to {FIGURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
