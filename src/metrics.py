"""Circuit and classification metric helpers.

Provides functions to compute classification performance metrics
(accuracy, precision, recall, F1, ROC-AUC, confusion matrix),
evaluate circuits under noisy AerSimulator execution, and format summary tables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a GitHub-flavored Markdown table without tabulate.

    Args:
        df: Pandas DataFrame to format.

    Returns:
        Formatted markdown table string.
    """
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        row_str = [str(val) for val in row.values]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines) + "\n"


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for binary classification.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted continuous probabilities or decision values for class 1.
        model_name: Identifier for the model.

    Returns:
        Dictionary of computed metrics and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    auc = np.nan
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = np.nan

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Balanced Accuracy": round(bal_acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4) if not np.isnan(auc) else "N/A",
        "Confusion Matrix": cm,
    }


def evaluate_noisy_circuit(
    transpiled_circuit_with_theta: QuantumCircuit,
    x_params: ParameterVector,
    X_test: np.ndarray,
    simulator: AerSimulator,
    shots: int = 4096,
    seed_simulator: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a transpiled VQC circuit under noisy AerSimulator execution.

    Args:
        transpiled_circuit_with_theta: Physical circuit with trained weights already bound,
            containing measurement operations and x_params unbound.
        x_params: ParameterVector for data inputs.
        X_test: Test feature matrix.
        simulator: Configured AerSimulator with processor noise model.
        shots: Number of measurement shots.
        seed_simulator: Random seed for simulation reproducibility.

    Returns:
        Tuple of (expectation_values, class_1_probabilities, binary_predictions).
    """
    bound_circuits = []
    for x_val in X_test:
        bind_dict = {x_params[i]: x_val[i] for i in range(len(x_params))}
        bound_circuits.append(transpiled_circuit_with_theta.assign_parameters(bind_dict))

    job = simulator.run(bound_circuits, shots=shots, seed_simulator=seed_simulator)
    result = job.result()
    all_counts = [result.get_counts(i) for i in range(len(bound_circuits))]

    expectations: List[float] = []
    probabilities: List[float] = []

    for counts in all_counts:
        exp = 0.0
        for bitstring, count in counts.items():
            bits = [int(c) for c in bitstring if c in ("0", "1")]
            parity = sum(bits) % 2
            sign = -1.0 if parity == 1 else 1.0
            exp += sign * count
        exp /= shots
        expectations.append(exp)
        p1 = (exp + 1.0) / 2.0
        probabilities.append(p1)

    exp_arr = np.array(expectations)
    prob_arr = np.array(probabilities)
    binary_preds = (exp_arr >= 0.0).astype(int)

    return exp_arr, prob_arr, binary_preds


def build_metrics_summary(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format a list of metric dictionaries into a clean summary DataFrame.

    Args:
        metrics_list: List of metric dictionaries from compute_classification_metrics.

    Returns:
        Clean summary DataFrame excluding complex objects like confusion matrices.
    """
    rows = []
    for m in metrics_list:
        row = {k: v for k, v in m.items() if k != "Confusion Matrix"}
        rows.append(row)
    return pd.DataFrame(rows)
