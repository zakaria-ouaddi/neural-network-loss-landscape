"""
Option A — Animated 3D GD visualisation (saved as GIF).

Produces rotating 3D surface with the GD path drawn step-by-step.
Each frame adds one more trajectory point + gradient arrow.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from mpl_toolkits.mplot3d import Axes3D  # noqa
from typing import List, Optional

from visualization import _interpolate_loss_on_grid


def create_gd_animation(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    Z: np.ndarray,
    coords: torch.Tensor,
    neg_gu: Optional[List[float]] = None,
    neg_gv: Optional[List[float]] = None,
    grad_indices: Optional[List[int]] = None,
    title: str = "Gradient Descent Animation",
    save_path: str = "figures/animation.gif",
    fps: int = 8,
    rotate: bool = True,
) -> None:
    """
    Create an animated GIF of GD traversing the loss surface.

    The camera slowly rotates while trajectory points appear one by one.

    Parameters
    ----------
    alpha_grid, beta_grid, Z : ndarray (res, res)
    coords : Tensor (T+1, 2)
    neg_gu, neg_gv : gradient projections (optional)
    grad_indices : which trajectory steps have gradients
    title : str
    save_path : str
    fps : int
    rotate : bool  — rotate camera across frames
    """
    coords_np = coords.numpy()
    loss_traj = _interpolate_loss_on_grid(coords_np, alpha_grid, beta_grid, Z)
    n_steps = len(coords_np)

    # Build gradient lookup: step_index → (gu, gv)
    grad_map = {}
    if neg_gu is not None and neg_gv is not None and grad_indices is not None:
        for k, idx in enumerate(grad_indices):
            grad_map[idx] = (neg_gu[k], neg_gv[k])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # We'll draw the surface once, then update trajectory line per frame
    ax.plot_surface(
        alpha_grid, beta_grid, Z,
        cmap=cm.viridis, alpha=0.6, edgecolor="none",
    )

    # Pre-create artists that we'll update
    (path_line,) = ax.plot([], [], [], "r-", linewidth=2.5, zorder=5)
    (path_dots,) = ax.plot([], [], [], "ro", markersize=4, zorder=5)
    start_marker = ax.scatter(
        [coords_np[0, 0]], [coords_np[0, 1]], [loss_traj[0]],
        c="lime", s=200, marker="*", edgecolors="k", linewidths=1.5, zorder=6,
    )

    ax.set_xlabel("PC1 (α)", fontsize=11)
    ax.set_ylabel("PC2 (β)", fontsize=11)
    ax.set_zlabel("Loss", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Total frames = trajectory steps + extra rotation at end
    extra_rotation_frames = 30
    total_frames = n_steps + extra_rotation_frames

    # Keep references to quiver arrows so we can remove old ones
    quiver_artists = []

    def update(frame):
        nonlocal quiver_artists

        # How many trajectory points to show
        t = min(frame, n_steps)

        # Update trajectory line
        path_line.set_data(coords_np[:t, 0], coords_np[:t, 1])
        path_line.set_3d_properties(loss_traj[:t])

        path_dots.set_data(coords_np[:t, 0], coords_np[:t, 1])
        path_dots.set_3d_properties(loss_traj[:t])

        # Remove old quiver arrows
        for q in quiver_artists:
            q.remove()
        quiver_artists.clear()

        # Draw gradient arrow at current step if available
        if t > 0 and (t - 1) in grad_map:
            gu, gv = grad_map[t - 1]
            scale = 0.15
            q = ax.quiver(
                coords_np[t - 1, 0], coords_np[t - 1, 1], loss_traj[t - 1],
                gu * scale, gv * scale, 0,
                color="yellow", arrow_length_ratio=0.3, linewidth=2,
            )
            quiver_artists.append(q)

        # Place end marker at current position
        if t > 1:
            # (we'll just add a fresh scatter each time — a bit wasteful but safe)
            pass

        # Rotate camera
        if rotate:
            elev = 25 + 10 * np.sin(2 * np.pi * frame / total_frames)
            azim = 220 + frame * (360 / total_frames)
            ax.view_init(elev=elev, azim=azim)

        return path_line, path_dots

    # Sample every N-th step so the animation isn't too long
    step = max(1, n_steps // 60)
    frame_list = list(range(0, n_steps, step)) + list(range(n_steps, total_frames))

    anim = animation.FuncAnimation(
        fig, update, frames=frame_list, interval=1000 // fps, blit=False,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer="pillow", fps=fps, dpi=100)
    print(f"  → animation saved to {save_path}")
    plt.close(fig)


def create_multi_seed_animation(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    Z: np.ndarray,
    all_coords: dict,
    title: str = "Multi-Seed GD Animation",
    save_path: str = "figures/multi_seed_animation.gif",
    fps: int = 8,
) -> None:
    """
    Animated GIF showing all seed trajectories being drawn simultaneously.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        alpha_grid, beta_grid, Z,
        cmap=cm.viridis, alpha=0.5, edgecolor="none",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_coords)))
    lines = {}
    all_loss_traj = {}

    for idx, (seed, coords) in enumerate(all_coords.items()):
        c = coords.numpy()
        lt = _interpolate_loss_on_grid(c, alpha_grid, beta_grid, Z)
        all_loss_traj[seed] = lt
        (line,) = ax.plot([], [], [], color=colors[idx], linewidth=2, alpha=0.8,
                          label=f"seed {seed}")
        lines[seed] = (line, c, lt)

    ax.set_xlabel("PC1 (α)")
    ax.set_ylabel("PC2 (β)")
    ax.set_zlabel("Loss")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.legend(fontsize=7, loc="upper right")

    max_len = max(len(c) for _, (_, c, _) in lines.items())
    extra_rotation = 30
    total_frames = max_len + extra_rotation

    step = max(1, max_len // 60)
    frame_list = list(range(0, max_len, step)) + list(range(max_len, total_frames))

    def update(frame):
        t = min(frame, max_len)
        for seed, (line, c, lt) in lines.items():
            end = min(t, len(c))
            line.set_data(c[:end, 0], c[:end, 1])
            line.set_3d_properties(lt[:end])

        elev = 25 + 10 * np.sin(2 * np.pi * frame / total_frames)
        azim = 220 + frame * (360 / total_frames)
        ax.view_init(elev=elev, azim=azim)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=frame_list, interval=1000 // fps, blit=False,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer="pillow", fps=fps, dpi=100)
    print(f"  → animation saved to {save_path}")
    plt.close(fig)
