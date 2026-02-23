# Project Implementation Map: What & Where

This document serves as a direct mapping of the professor's assignment requirements to their exact location and implementation within our finalized notebook: `submission/Complete_Gradient_Descent_Visualization.ipynb`.

## 1. Core Requirements

| Requirement | Implementation Detail | Location in Notebook |
| :--- | :--- | :--- |
| **Implement Logistic Regression model** | `LogisticRegression(nn.Module)`: A simple 1-linear layer model with a Sigmoid activation. | Cell 8 |
| **Use `make_classification` dataset** | Generated 1000 samples, 20 features, 2 classes. | Cell 7 |
| **Train model & save parameter trajectory** | `train_model` loop saves `get_flat_params(model)` every epoch. | Cell 3 (Helper fx) & Cell 8 |
| **Apply PCA to the trajectory** | `pca_project` function centers weights and applies `sklearn.decomposition.PCA(n_components=2)`. | Cell 3 (Helper fx) & Cell 9 |
| **2D plot of the path in PCA space** | Trajectories are plotted onto the `(PC1, PC2)` grid. | Cell 13 (Subplots 1 & 3) |
| **Calculate Loss Surface radially** | `compute_loss_surface` evaluates model over an $\alpha$ (PC1) and $\beta$ (PC2) parameter grid. | Cell 3 (Helper fn) & Cell 10 |
| **Plot Loss Surface (3D / Contour)** | Plotly's `Surface` and `Contour` traces map the `logistic_loss_surface_grid`. | Cell 13 (Subplots 2 & 4), Cell 14 (3D) |
| **Calculate Gradients on the surface** | `compute_gradients_on_surface` calculates numerical gradients across the grid. | Cell 4 |
| **Combine Path, Surface & Gradients** | `visualize_loss_surface_with_gradients` (Convex) and `plot_landscape_dashboard` overlay all three elements. | Cell 13 & 25 |

---

## 2. Additional Task: Non-Convex Landscape

| Requirement | Implementation Detail | Location in Notebook |
| :--- | :--- | :--- |
| **Implement 2-Layer Neural Network** | `SimpleNN(nn.Module)`: A 2-layer MLP with a hidden layer of 32 units and ReLU activation. | Cell 20 |
| **Train from multiple initialization points** | A loop trains `SimpleNN` on `seeds = [42, 10, 123, 77, 999]`, storing distinct path histories. | Cell 20 |
| **Visualize multiple descents on one surface** | Computes a global PCA/Isomap basis covering *all* points, evaluating surface, and plotting 5 colored path lines. | Cell 25 & 28 |

---

## 3. Additional Task: Visualization Comparisons

| Requirement | Implementation Detail | Location in Notebook |
| :--- | :--- | :--- |
| **Compare different projection algorithms** | Implemented `isomap_project` using `sklearn.manifold.Isomap`. | Cell 25 & 28 |
| **Analyze projection artifacts** | Side-by-side plot comparing the PCA projection (floating/clipping) vs Isomap projection (adhering to manifold). | Cell 28 (Side-by-side) |
| **Isomap Surface evaluation** | Used `GridSearchCV` and `KNeighborsRegressor` to inversely interpolate the Isomap basis back to parameter space to evaluate loss. | Cell 26 |

---

## 4. Additional Task: Hessian Analysis

| Requirement | Implementation Detail | Location in Notebook |
| :--- | :--- | :--- |
| **Calculate Hessian over time** | Implemented `hessian_vector_product` and Power Iteration to find $\lambda_{max}$ without instantiating the full $N \times N$ matrix. | Cell 3 |
| **Plot Eigenvalue Trajectory** | A trajectory plot colored by the Hessian $\lambda_{max}$, proving the model moves from sharp cliffs to flat valleys. | Cell 31 |

---

## 5. Additional Task: Advanced Visualization

| Requirement | Implementation Detail | Location in Notebook |
| :--- | :--- | :--- |
| **Create interactive 3D visualizations** | All static charts replaced with `plotly.graph_objects` to enable panning, rotating, and zooming natively in the notebook. exported as standalone `.html` | Throughout (Cells 13, 14, 23, 25, 28) |
| **Create Animations (GIFs)** | Implemented a custom `update_plot` logic feeding into `matplotlib.animation.FuncAnimation` to chronologically trace the optimizer’s descent. | Cells 17, 28, 41 |

---

## 6. Originality: The 2D Micro-Model

| Contribution | Implementation Detail | Location in Notebook |
| :--- | :--- | :--- |
| **Total Artifact Elimination** | To avoid the distortion of *compressing* weights (PCA/Isomap), we engineered `TwoParamNN`, a network with exactly 2 parameters (`w1`, `w2`), allowing for a perfect 100% variance 2D mapping. | Cell 36 |
| **Compare Optimizers (SGD vs. Adam)** | Trained `TwoParamNN` using SGD and Adam from the exact same violent cliff position, capturing their respective histories. | Cell 37 |
| **Prove Optimizer Efficiency** | An animated plot showing SGD wildly oscillating due to steep gradients vs. Adam smoothly slicing through the valley via momentum. | Cell 41 |
