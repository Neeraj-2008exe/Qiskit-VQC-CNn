"""Coupling-map and benchmark plotting helpers.

Provides functions to generate publication-grade figures for processor topologies,
dataset distributions, circuit diagrams, training convergence, confusion matrices,
ROC curves, transpilation trade-offs, and classification performance.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from sklearn.metrics import roc_curve


# Clean aesthetic style defaults
plt.rcParams.update(
    {
        "font.sans-serif": "Arial",
        "font.family": "sans-serif",
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": False,
    }
)


def plot_coupling_map(
    coupling_map: List[List[int]],
    num_qubits: int,
    title: str = "Coupling Map",
    save_path: Optional[str] = None,
    node_color: str = "#4A90E2",
    edge_color: str = "#2C3E50",
) -> plt.Figure:
    """Plot processor coupling topology using NetworkX and Matplotlib.

    Args:
        coupling_map: List of [source, target] qubit pairs.
        num_qubits: Total number of qubits on the processor.
        title: Figure title.
        save_path: Optional path to save PNG figure.
        node_color: Hex color for qubit nodes.
        edge_color: Hex color for coupling links.

    Returns:
        Matplotlib Figure object.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_qubits))
    for u, v in coupling_map:
        G.add_edge(u, v)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Layout selection: linear for line graphs, planar/spring for 2D meshes
    if num_qubits == 5 and all(abs(u - v) == 1 for u, v in coupling_map):
        pos = {i: (i * 1.5, 0) for i in range(num_qubits)}
    else:
        # Custom structured layout for Processor B (heavy-hex inspired mesh)
        pos = nx.spring_layout(G, seed=42, k=1.2)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_color,
        node_size=1100,
        edgecolors="#1A252F",
        linewidths=2.0,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=11,
        font_weight="bold",
        font_color="#FFFFFF",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_color,
        width=2.5,
        alpha=0.85,
    )

    ax.set_title(f"{title} (N = {num_qubits} Qubits, |E| = {len(coupling_map)} Couplings)", pad=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_dataset_distribution(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot feature distributions split by binary class label.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target vector.
        feature_names: Optional names for each feature.
        save_path: Optional path to save PNG figure.

    Returns:
        Matplotlib Figure object.
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i} (informative)" if i < 3 else f"Feature {i} (redundant)" for i in range(n_features)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    axes = axes.flatten()

    c0_color = "#3498DB"  # Blue
    c1_color = "#E74C3C"  # Coral / Red

    for i in range(n_features):
        ax = axes[i]
        vals_0 = X[y == 0, i]
        vals_1 = X[y == 1, i]

        ax.hist(vals_0, bins=15, alpha=0.6, color=c0_color, label="Class 0 (y=0)", density=True)
        ax.hist(vals_1, bins=15, alpha=0.6, color=c1_color, label="Class 1 (y=1)", density=True)
        ax.set_title(feature_names[i], fontweight="bold")
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle("PS-Q1 Synthetic Dataset Feature Distributions (N=200, 4 Features)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_loss(
    loss_history: List[float],
    title: str = "VQC Parameter Optimization Trajectory (COBYLA)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot convergence curve of VQC training cost function.

    Args:
        loss_history: List of loss values over optimizer evaluations.
        title: Figure title.
        save_path: Optional path to save PNG figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    evals = range(1, len(loss_history) + 1)

    ax.plot(evals, loss_history, color="#2ECC71", lw=2.2, label="Mean Squared Error Loss")
    ax.scatter(evals[0], loss_history[0], color="#E74C3C", s=60, zorder=5, label=f"Initial: {loss_history[0]:.4f}")
    min_idx = int(np.argmin(loss_history))
    ax.scatter(
        evals[min_idx],
        loss_history[min_idx],
        color="#8E44AD",
        s=80,
        marker="*",
        zorder=5,
        label=f"Minimum: {loss_history[min_idx]:.4f} (eval {evals[min_idx]})",
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Objective Function Evaluations")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrices(
    cms_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot side-by-side confusion matrices for all evaluated models.

    Args:
        cms_dict: Dictionary mapping model names to 2x2 confusion matrices.
        save_path: Optional path to save PNG figure.

    Returns:
        Matplotlib Figure object.
    """
    models = list(cms_dict.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), dpi=300)
    if n == 1:
        axes = [axes]

    cmap = plt.cm.Blues

    for idx, name in enumerate(models):
        ax = axes[idx]
        cm = cms_dict[name]
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(name, fontweight="bold", pad=10, fontsize=11)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["True 0", "True 1"])

        # Annotate cells with counts and percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100.0
                text_color = "white" if val > cm.max() / 2.0 else "black"
                ax.text(
                    j,
                    i,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=10,
                )

        ax.set_ylabel("True Label" if idx == 0 else "")
        ax.set_xlabel("Predicted Label")

    fig.suptitle("Test Set Confusion Matrix Comparison (N = 40 samples)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot overlaid ROC curves for baseline and quantum models.

    Args:
        roc_data: Dictionary mapping model name to tuple of (fpr, tpr, roc_auc).
        save_path: Optional path to save PNG figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

    palette = {
        "Classical Baseline (Logistic Regression)": "#7F8C8D",
        "Ideal VQC (Statevector)": "#2980B9",
        "Processor A (Noisy 5Q Linear)": "#E67E22",
        "Processor B (Noisy 9Q Mesh)": "#27AE60",
    }

    for name, (fpr, tpr, auc_val) in roc_data.items():
        color = palette.get(name, "#8E44AD")
        label_text = f"{name} (AUC = {auc_val:.3f})" if not np.isnan(auc_val) else f"{name} (AUC = N/A)"
        ax.plot(fpr, tpr, lw=2.2, color=color, label=label_text)

    # Reference random guessing diagonal
    ax.plot([0, 1], [0, 1], color="#BDC3C7", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_transpilation_comparison(
    transpilation_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot bar chart comparing depth, total ops, and CX counts across processors.

    Args:
        transpilation_df: DataFrame output from build_transpilation_summary.
        save_path: Optional path to save PNG figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    names = transpilation_df["Name"].tolist()
    x = np.arange(len(names))
    width = 0.35

    # Subplot 1: Depth and Total Operations
    ax1.bar(x - width / 2, transpilation_df["Depth"], width, label="Circuit Depth", color="#3498DB")
    ax1.bar(x + width / 2, transpilation_df["Total Operations"], width, label="Total Operations", color="#9B59B6")
    ax1.set_ylabel("Count")
    ax1.set_title("Circuit Depth & Total Operations", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.split(" ")[0] + (" " + n.split(" ")[1] if len(n.split(" ")) > 1 else "") for n in names], rotation=15)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(frameon=True)

    # Subplot 2: CX Count and Extra Routing CX
    ax2.bar(x - width / 2, transpilation_df["CX Count"], width, label="Physical CX Count", color="#E67E22")
    ax2.bar(x + width / 2, transpilation_df["Extra CX (Routing)"], width, label="Routing CX Overhead", color="#E74C3C")
    ax2.set_ylabel("Gate Count")
    ax2.set_title("CX Count & Routing Overhead", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.split(" ")[0] + (" " + n.split(" ")[1] if len(n.split(" ")) > 1 else "") for n in names], rotation=15)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(frameon=True)

    fig.suptitle("Transpilation Structural Metrics: Logical vs. Processor A vs. Processor B", fontweight="bold", fontsize=13)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot grouped bar chart comparing classification metrics across all models.

    Args:
        metrics_df: DataFrame with model names and columns for Accuracy, Precision, Recall, F1 Score.
        save_path: Optional path to save PNG figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    metric_cols = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(metrics_df))
    n_metrics = len(metric_cols)
    width = 0.15

    colors = ["#2980B9", "#1ABC9C", "#F39C12", "#E74C3C", "#8E44AD"]

    for i, col in enumerate(metric_cols):
        offsets = x + (i - (n_metrics - 1) / 2) * width
        vals = metrics_df[col].astype(float)
        ax.bar(offsets, vals, width, label=col, color=colors[i % len(colors)])

    ax.set_ylim([0.0, 1.1])
    ax.set_ylabel("Score")
    ax.set_title("Classification Performance Across Architectures and Baseline", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"], rotation=15, ha="right", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="lower right", frameon=True, ncol=3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def save_circuit_diagram_image(
    circuit: QuantumCircuit,
    title: str = "Quantum Circuit Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render circuit text representation into a high-DPI image.

    Args:
        circuit: QuantumCircuit to render.
        title: Figure title.
        save_path: Optional path to save PNG figure.

    Returns:
        Matplotlib Figure object.
    """
    circuit_str = str(circuit.draw(output="text"))

    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.axis("off")
    ax.text(
        0.02,
        0.5,
        circuit_str,
        fontfamily="monospace",
        fontsize=8.5,
        va="center",
        color="#2C3E50",
    )
    ax.set_title(title, fontweight="bold", fontsize=12, pad=15)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
