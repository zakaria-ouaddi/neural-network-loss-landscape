"""
Option B — Interactive 3D visualisations via Plotly.

Generates self-contained .html files that can be opened in any browser.
The professor can rotate, zoom, pan, and hover over trajectory points.
"""

import os
import numpy as np
import torch
import plotly.graph_objects as go
from typing import List, Optional, Dict

from visualization import _interpolate_loss_on_grid


def create_interactive_surface(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    Z: np.ndarray,
    coords: torch.Tensor,
    neg_gu: Optional[List[float]] = None,
    neg_gv: Optional[List[float]] = None,
    grad_indices: Optional[List[int]] = None,
    title: str = "Interactive Loss Surface",
    save_path: str = "figures/interactive_surface.html",
) -> None:
    """
    Interactive Plotly 3D surface with GD trajectory.

    Parameters
    ----------
    alpha_grid, beta_grid, Z : ndarray (res, res)
    coords : Tensor (T+1, 2)
    neg_gu, neg_gv, grad_indices : gradient arrows (optional)
    title : str
    save_path : str  — output .html file
    """
    coords_np = coords.numpy()
    loss_traj = _interpolate_loss_on_grid(coords_np, alpha_grid, beta_grid, Z)

    fig = go.Figure()

    # ── Loss surface ──────────────────────────────────────────────────────
    fig.add_trace(go.Surface(
        x=alpha_grid,
        y=beta_grid,
        z=Z,
        colorscale="Viridis",
        opacity=0.75,
        showscale=True,
        colorbar=dict(title="Loss", x=1.05),
        name="Loss Surface",
        hoverinfo="skip",
    ))

    # ── GD trajectory ─────────────────────────────────────────────────────
    # Hover text with epoch + loss
    hover_text = [
        f"Epoch {i}<br>α={coords_np[i,0]:.3f}<br>β={coords_np[i,1]:.3f}<br>Loss={loss_traj[i]:.4f}"
        for i in range(len(coords_np))
    ]

    fig.add_trace(go.Scatter3d(
        x=coords_np[:, 0],
        y=coords_np[:, 1],
        z=loss_traj,
        mode="lines+markers",
        marker=dict(size=3, color="red"),
        line=dict(width=4, color="red"),
        name="GD Path",
        text=hover_text,
        hoverinfo="text",
    ))

    # ── Start / end markers ───────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[coords_np[0, 0]],
        y=[coords_np[0, 1]],
        z=[loss_traj[0]],
        mode="markers+text",
        marker=dict(size=10, color="lime", symbol="diamond",
                    line=dict(width=2, color="black")),
        text=["Start"],
        textposition="top center",
        name="Start",
        hoverinfo="text",
        hovertext=f"Start<br>Loss={loss_traj[0]:.4f}",
    ))

    fig.add_trace(go.Scatter3d(
        x=[coords_np[-1, 0]],
        y=[coords_np[-1, 1]],
        z=[loss_traj[-1]],
        mode="markers+text",
        marker=dict(size=10, color="red", symbol="diamond",
                    line=dict(width=2, color="black")),
        text=["End"],
        textposition="top center",
        name="End",
        hoverinfo="text",
        hovertext=f"End<br>Loss={loss_traj[-1]:.4f}",
    ))

    # ── Gradient arrows (as Cone traces) ──────────────────────────────────
    if neg_gu is not None and neg_gv is not None and grad_indices is not None:
        gx, gy, gz = [], [], []
        gu_list, gv_list, gw_list = [], [], []
        for k, idx in enumerate(grad_indices):
            if idx < len(coords_np):
                gx.append(coords_np[idx, 0])
                gy.append(coords_np[idx, 1])
                gz.append(loss_traj[idx])
                gu_list.append(neg_gu[k])
                gv_list.append(neg_gv[k])
                gw_list.append(0)

        if gx:
            fig.add_trace(go.Cone(
                x=gx, y=gy, z=gz,
                u=gu_list, v=gv_list, w=gw_list,
                sizemode="absolute",
                sizeref=0.15,
                colorscale=[[0, "orange"], [1, "orange"]],
                showscale=False,
                name="Gradient",
                opacity=0.7,
            ))

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title="PC1 (α)",
            yaxis_title="PC2 (β)",
            zaxis_title="Loss",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.6),
        ),
        width=1000,
        height=800,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path, include_plotlyjs=True)
    print(f"  → interactive plot saved to {save_path}")


def create_interactive_multi_seed(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    Z: np.ndarray,
    all_coords: Dict[int, torch.Tensor],
    title: str = "Interactive Multi-Seed Comparison",
    save_path: str = "figures/interactive_multi_seed.html",
) -> None:
    """
    Interactive Plotly 3D surface with multiple seed trajectories.
    Each seed can be toggled on/off via the legend.
    """
    import plotly.express as px

    fig = go.Figure()

    # Surface
    fig.add_trace(go.Surface(
        x=alpha_grid, y=beta_grid, z=Z,
        colorscale="Viridis", opacity=0.6, showscale=True,
        colorbar=dict(title="Loss", x=1.05),
        name="Loss Surface", hoverinfo="skip",
    ))

    # One trace per seed
    colors = px.colors.qualitative.T10
    for idx, (seed, coords) in enumerate(all_coords.items()):
        c = coords.numpy()
        lt = _interpolate_loss_on_grid(c, alpha_grid, beta_grid, Z)
        color = colors[idx % len(colors)]

        hover = [
            f"Seed {seed} • Epoch {i}<br>Loss={lt[i]:.4f}"
            for i in range(len(c))
        ]

        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=lt,
            mode="lines+markers",
            marker=dict(size=2, color=color),
            line=dict(width=3, color=color),
            name=f"Seed {seed}",
            text=hover,
            hoverinfo="text",
        ))

        # End marker
        fig.add_trace(go.Scatter3d(
            x=[c[-1, 0]], y=[c[-1, 1]], z=[lt[-1]],
            mode="markers",
            marker=dict(size=7, color=color, symbol="diamond",
                        line=dict(width=1.5, color="black")),
            name=f"Seed {seed} end",
            showlegend=False,
            hovertext=f"Seed {seed} final<br>Loss={lt[-1]:.4f}",
            hoverinfo="text",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title="PC1 (α)",
            yaxis_title="PC2 (β)",
            zaxis_title="Loss",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.6),
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50),
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path, include_plotlyjs=True)
    print(f"  → interactive plot saved to {save_path}")
