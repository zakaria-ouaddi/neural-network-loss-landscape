"""
Phase VIII — Steps 8.1–8.3: 3-D loss surface visualisation.
Phase IX  — Step 9.2 : PCA vs Isomap side-by-side comparison.

All public functions save figures to *save_dir* and call plt.show().
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for 3d projection
from typing import List, Optional, Dict


# ═══════════════════════════════════════════════════════════════════════════
# Step 8.1 + 8.2 — Single-trajectory 3-D surface + contour + gradient quiver
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_surface_3d(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    Z: np.ndarray,
    coords: torch.Tensor,
    neg_gu: Optional[List[float]] = None,
    neg_gv: Optional[List[float]] = None,
    grad_indices: Optional[List[int]] = None,
    title: str = "Loss Surface",
    save_path: Optional[str] = None,
) -> None:
    """
    Three-panel figure:
      1. 3-D surface with GD trajectory
      2. Contour map with trajectory + gradient quiver arrows
      3. Loss vs iteration

    Parameters
    ----------
    alpha_grid, beta_grid, Z : ndarray, shape (res, res)
    coords : Tensor, shape (T+1, 2)
    neg_gu, neg_gv : list[float] (optional)
        Negative gradient projections for quiver.
    grad_indices : list[int] (optional)
        Which trajectory steps correspond to the gradient entries.
    title : str
    save_path : str or None
    """
    coords_np = coords.numpy()
    fig = plt.figure(figsize=(20, 6))

    # ── Panel 1: 3-D surface ──────────────────────────────────────────────
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(
        alpha_grid, beta_grid, Z,
        cmap=cm.viridis, alpha=0.7, edgecolor="none",
    )

    # Interpolate loss along trajectory
    loss_traj = _interpolate_loss_on_grid(coords_np, alpha_grid, beta_grid, Z)

    ax1.plot(
        coords_np[:, 0], coords_np[:, 1], loss_traj,
        "r-o", linewidth=2, markersize=3, zorder=5, label="GD path",
    )
    ax1.scatter(*coords_np[0], loss_traj[0], c="lime", s=180, marker="*",
                edgecolors="k", linewidths=1.5, zorder=6, label="Start")
    ax1.scatter(*coords_np[-1], loss_traj[-1], c="red", s=180, marker="*",
                edgecolors="k", linewidths=1.5, zorder=6, label="End")

    ax1.set_xlabel("PC1 (α)")
    ax1.set_ylabel("PC2 (β)")
    ax1.set_zlabel("Loss")
    ax1.set_title(title, fontweight="bold")
    ax1.legend(fontsize=8)

    # ── Panel 2: Contour + quiver ─────────────────────────────────────────
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(alpha_grid, beta_grid, Z, levels=40, cmap=cm.viridis)
    fig.colorbar(contour, ax=ax2)

    ax2.plot(coords_np[:, 0], coords_np[:, 1], "r-o", linewidth=2, markersize=3)
    ax2.scatter(*coords_np[0], c="lime", s=180, marker="*", edgecolors="k",
                linewidths=1.5, zorder=5)
    ax2.scatter(*coords_np[-1], c="red", s=180, marker="*", edgecolors="k",
                linewidths=1.5, zorder=5)

    if neg_gu is not None and neg_gv is not None and grad_indices is not None:
        for k, t in enumerate(grad_indices):
            if t < len(coords_np):
                ax2.quiver(
                    coords_np[t, 0], coords_np[t, 1],
                    neg_gu[k], neg_gv[k],
                    angles="xy", scale_units="xy", scale=5,
                    color="yellow", alpha=0.7, width=0.005,
                )

    ax2.set_xlabel("PC1 (α)")
    ax2.set_ylabel("PC2 (β)")
    ax2.set_title("Contour + Gradient Arrows", fontweight="bold")
    ax2.grid(alpha=0.3)

    # ── Panel 3: Loss curve ───────────────────────────────────────────────
    ax3 = fig.add_subplot(133)
    ax3.plot(loss_traj, "b-", linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss Convergence", fontweight="bold")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  → saved {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Step 8.3 — Multiple seeds on the same surface
# ═══════════════════════════════════════════════════════════════════════════

def plot_multiple_seeds(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    Z: np.ndarray,
    all_coords: Dict[int, torch.Tensor],
    title: str = "Multiple Seeds",
    save_path: Optional[str] = None,
) -> None:
    """
    Overlay trajectories from multiple seeds.

    Parameters
    ----------
    alpha_grid, beta_grid, Z : ndarray
    all_coords : dict[seed → Tensor(T+1, 2)]
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_coords)))

    fig = plt.figure(figsize=(20, 6))

    # ── Panel 1: 3-D ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(alpha_grid, beta_grid, Z, cmap=cm.viridis, alpha=0.5,
                     edgecolor="none")

    for idx, (seed, coords) in enumerate(all_coords.items()):
        c = coords.numpy()
        lt = _interpolate_loss_on_grid(c, alpha_grid, beta_grid, Z)
        ax1.plot(c[:, 0], c[:, 1], lt, color=colors[idx], linewidth=2,
                 alpha=0.8, label=f"seed {seed}")
        ax1.scatter(c[-1, 0], c[-1, 1], lt[-1], color=colors[idx], s=120,
                    marker="*", edgecolors="k", linewidths=1)

    ax1.set_xlabel("PC1 (α)")
    ax1.set_ylabel("PC2 (β)")
    ax1.set_zlabel("Loss")
    ax1.set_title(title, fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")

    # ── Panel 2: Contour ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(132)
    ax2.contourf(alpha_grid, beta_grid, Z, levels=40, cmap=cm.viridis)

    for idx, (seed, coords) in enumerate(all_coords.items()):
        c = coords.numpy()
        ax2.plot(c[:, 0], c[:, 1], color=colors[idx], linewidth=2, alpha=0.8)
        ax2.scatter(c[-1, 0], c[-1, 1], color=colors[idx], s=120, marker="*",
                    edgecolors="k", linewidths=1, zorder=5)

    ax2.set_xlabel("PC1 (α)")
    ax2.set_ylabel("PC2 (β)")
    ax2.set_title("Contour — All Seeds", fontweight="bold")
    ax2.grid(alpha=0.3)

    # ── Panel 3: Convergence ──────────────────────────────────────────────
    ax3 = fig.add_subplot(133)
    for idx, (seed, coords) in enumerate(all_coords.items()):
        c = coords.numpy()
        lt = _interpolate_loss_on_grid(c, alpha_grid, beta_grid, Z)
        ax3.plot(lt, color=colors[idx], linewidth=1.5, alpha=0.7, label=f"seed {seed}")

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Convergence — All Seeds", fontweight="bold")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  → saved {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Phase IX — Step 9.2:  PCA vs Isomap side-by-side
# ═══════════════════════════════════════════════════════════════════════════

def plot_pca_vs_isomap(
    pca_coords: torch.Tensor,
    isomap_coords: np.ndarray,
    pca_var: np.ndarray,
    iso_residual: float,
    title: str = "PCA vs Isomap",
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side trajectory comparison.

    Parameters
    ----------
    pca_coords : Tensor (T+1, 2)
    isomap_coords : ndarray (T+1, 2)
    pca_var : ndarray, shape (2,)
    iso_residual : float
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA
    pc = pca_coords.numpy()
    axes[0].plot(pc[:, 0], pc[:, 1], "o-", linewidth=2, markersize=3)
    axes[0].scatter(*pc[0], c="lime", s=180, marker="*", edgecolors="k",
                    linewidths=1.5, zorder=5, label="Start")
    axes[0].scatter(*pc[-1], c="red", s=180, marker="*", edgecolors="k",
                    linewidths=1.5, zorder=5, label="End")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title(
        f"PCA  (var explained: {pca_var.sum():.3f})", fontweight="bold"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Isomap
    axes[1].plot(isomap_coords[:, 0], isomap_coords[:, 1], "o-", linewidth=2,
                 markersize=3, color="tab:orange")
    axes[1].scatter(*isomap_coords[0], c="lime", s=180, marker="*",
                    edgecolors="k", linewidths=1.5, zorder=5, label="Start")
    axes[1].scatter(*isomap_coords[-1], c="red", s=180, marker="*",
                    edgecolors="k", linewidths=1.5, zorder=5, label="End")
    axes[1].set_xlabel("Isomap-1")
    axes[1].set_ylabel("Isomap-2")
    axes[1].set_title(
        f"Isomap  (residual var: {iso_residual:.4f})", fontweight="bold"
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  → saved {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Phase X — Step 10.3: Curvature-coloured trajectory
# ═══════════════════════════════════════════════════════════════════════════

def plot_curvature_trajectory(
    coords: torch.Tensor,
    eigenvalues: List[float],
    title: str = "Curvature-Coloured Trajectory",
    save_path: Optional[str] = None,
) -> None:
    """
    2-D trajectory where each segment is coloured by Hessian top eigenvalue.

    Parameters
    ----------
    coords : Tensor, shape (T+1, 2)
    eigenvalues : list[float], length T+1
    """
    c = coords.numpy()
    fig, ax = plt.subplots(figsize=(8, 6))

    eig_arr = np.array(eigenvalues)
    norm = plt.Normalize(vmin=eig_arr.min(), vmax=eig_arr.max())
    cmap = cm.coolwarm

    for i in range(len(c) - 1):
        color = cmap(norm(eig_arr[i]))
        ax.plot(c[i:i+2, 0], c[i:i+2, 1], color=color, linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Top Hessian Eigenvalue")

    ax.scatter(*c[0], c="lime", s=180, marker="*", edgecolors="k",
               linewidths=1.5, zorder=5, label="Start")
    ax.scatter(*c[-1], c="red", s=180, marker="*", edgecolors="k",
               linewidths=1.5, zorder=5, label="End")

    ax.set_xlabel("PC1 (α)")
    ax.set_ylabel("PC2 (β)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  → saved {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _interpolate_loss_on_grid(
    coords_np: np.ndarray,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    For each (α, β) coordinate find nearest grid cell and return loss.
    """
    loss_traj = np.zeros(len(coords_np))
    for k in range(len(coords_np)):
        i = np.argmin(np.abs(beta_grid[:, 0] - coords_np[k, 1]))
        j = np.argmin(np.abs(alpha_grid[0, :] - coords_np[k, 0]))
        i = np.clip(i, 0, Z.shape[0] - 1)
        j = np.clip(j, 0, Z.shape[1] - 1)
        loss_traj[k] = Z[i, j]
    return loss_traj
