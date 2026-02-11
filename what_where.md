# What & Where — Professor's Tasks → Implementation

| # | Professor's Task Description | Where It's Implemented | Key Functions / Classes |
|---|-----|------|------|
| **1** | Implement a neural network in PyTorch that performs logistic regression. Train on simulated data using `make_classification`. | [models.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/models.py) — model definition | `LogisticRegressionModel` (line 27) |
| | | [data.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/data.py) — data generation | `generate_data()` uses `make_classification` |
| | | [training.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/training.py) — training loop | `train()` (line 33) |
| | | [main.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/main.py) — orchestration | `run_logistic()` (line 55) |
| **2** | Train the model, collect frequent weight checkpoints. Apply PCA to reduce weight dimensionality to 2. | [training.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/training.py) — checkpoint collection | `train()` appends `get_flat_params()` every epoch → `trajectory` list |
| | | [projection.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/projection.py) — PCA projection | `pca_project()` (line 30): centres at θ*, fits PCA, extracts u/v |
| | | [filter_norm.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/filter_norm.py) — direction normalization | `filter_normalize_uv()` called inside `pca_project()` before projection |
| **3** | Implement 3D visualization of loss surface and gradient for logistic regression, updating at each GD step to show gradient directions. | [loss_surface.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/loss_surface.py) — surface computation | `compute_loss_surface()`: sweeps θ* + αu + βv grid |
| | | [gradients.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/gradients.py) — gradient arrows | `compute_projected_gradients()`: projects ∇L onto u, v |
| | | [visualization.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/visualization.py) — 3D plot | `plot_loss_surface_3d()`: surface + trajectory + quiver arrows |
| | | **Output** | `figures/logistic_3d.png` |
| **4** | Replace logistic regression with a simple two-layer neural network (non-convex). Analyse gradient descent behaviour. | [models.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/models.py) — MLP model | `MLP` class (line 63): Linear → ReLU → Linear |
| | | [main.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/main.py) — MLP experiment | `run_mlp_single()` (line 96) |
| | | [hessian.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/hessian.py) — curvature analysis | `compute_eigenvalues_along_trajectory()` for GD behaviour analysis |
| | | **Output** | `figures/mlp_3d.png`, `figures/mlp_curvature.png` |
| **5** | Retrain with different starting seeds. Visualise multiple paths (to potentially different local minima) in the same image. | [training.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/training.py) — multi-seed | `train_multiple_seeds()` (line 97): loops over `seed_list` |
| | | [visualization.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/visualization.py) — overlay plot | `plot_multiple_seeds()`: all 10 paths on same surface |
| | | [main.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/main.py) — orchestration | `run_mlp_multi_seed()` (line 142) |
| | | **Output** | `figures/mlp_multi_seed.png` |
| **6** | Research and implement a different dimensionality reduction method. Compare resulting figures to original PCA. | [projection.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/projection.py) — Isomap | `isomap_project()` (line 100): non-linear manifold method |
| | | [visualization.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/visualization.py) — comparison | `plot_pca_vs_isomap()`: side-by-side with variance metrics |
| | | **Output** | `figures/logistic_pca_vs_isomap.png`, `figures/mlp_pca_vs_isomap.png` |

## Beyond Requirements (extra credit)

| Enhancement | Where |
|-------------|-------|
| Filter normalization (Li et al. 2018) | [filter_norm.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/filter_norm.py) applied inside [projection.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/projection.py) |
| Hessian analysis (curvature along trajectory) | [hessian.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/hessian.py) → `figures/mlp_curvature.png` |
| Optimizer comparison (SGD / Momentum / Adam) | `run_optimizer_comparison()` in [main.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/main.py) → `figures/optimizer_comparison.png` |
| Animated 3D GD visualisation (GIF) | [animations.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/animations.py) |
| Interactive 3D HTML (Plotly) | [interactive_plots.py](file:///home/zakaria/Desktop/ML%20Bremen/Advanced-Machine-Learning/AML_Project/interactive_plots.py) |
