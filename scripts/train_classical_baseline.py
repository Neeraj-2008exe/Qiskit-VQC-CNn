"""Leakage-safe Classical Logistic Regression Baseline for PS-Q1: ECG Biosignal Classification.

Builds and evaluates a StandardScaler -> LogisticRegression pipeline on the official
QML-BIO-MITBIH ECG dataset (30 features, binary target: 0=Normal, 1=Arrhythmia).
Strictly adheres to QFF2026 data integrity and no-leakage guidelines:
- Fit scaler ONLY on training data (data/q1/ecg_train.csv).
- Evaluate ONLY on validation data (data/q1/ecg_val.csv) for hyperparameter selection.
- Test features (data/q1/ecg_test_features.csv) are NEVER loaded or used for training/tuning.
- Metric: macro-averaged F1 (f1_score(y_true, y_pred, average="macro")).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

# Exactly the 30 official statistical features defined in schemas/q1_feature_dictionary.md
EXPECTED_FEATURES: List[str] = [
    "w1_mean", "w1_std", "w1_min", "w1_max", "w1_skew", "w1_kurt",
    "w2_mean", "w2_std", "w2_min", "w2_max", "w2_skew", "w2_kurt",
    "w3_mean", "w3_std", "w3_min", "w3_max", "w3_skew", "w3_kurt",
    "w4_mean", "w4_std", "w4_min", "w4_max", "w4_skew", "w4_kurt",
    "w5_mean", "w5_std", "w5_min", "w5_max", "w5_skew", "w5_kurt",
]


def resolve_project_root(custom_root: Optional[str] = None) -> Path:
    """Resolve the canonical project root path deterministically.

    Args:
        custom_root: Optional path provided via CLI.

    Returns:
        Resolved absolute Path to project root.
    """
    if custom_root:
        root = Path(custom_root).resolve()
        if (root / "data" / "q1" / "ecg_train.csv").exists():
            return root

    # Check current working directory
    cwd = Path.cwd().resolve()
    if (cwd / "data" / "q1" / "ecg_train.csv").exists():
        return cwd

    # Check script's parent directory (scripts/ -> root)
    script_parent = Path(__file__).resolve().parent.parent
    if (script_parent / "data" / "q1" / "ecg_train.csv").exists():
        return script_parent

    raise FileNotFoundError(
        "Could not resolve project root containing data/q1/ecg_train.csv. "
        "Please provide --root <path_to_QFF2026_QML_Challenge_Kit>."
    )


def run_baseline_grid(
    train_path: Path,
    val_path: Path,
    c_values: Optional[List[float]] = None,
    class_weights: Optional[List[Optional[str]]] = None,
    max_iter: int = 5000,
    random_state: int = 42,
    solver: str = "lbfgs",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train and evaluate the logistic regression grid on official ECG partitions.

    Args:
        train_path: Path to ecg_train.csv.
        val_path: Path to ecg_val.csv.
        c_values: Regularization inverse strengths.
        class_weights: List of class_weight options (None, 'balanced').
        max_iter: Maximum solver iterations.
        random_state: Deterministic random seed.
        solver: Optimization algorithm (lbfgs).

    Returns:
        Tuple of (results_dataframe, best_configuration_dict).
    """
    if c_values is None:
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    if class_weights is None:
        class_weights = [None, "balanced"]

    print("==================================================================")
    print("   QFF2026 PS-Q1: CLASSICAL LOGISTIC REGRESSION BASELINE         ")
    print("==================================================================")
    print(f"Loading training data:   {train_path}")
    print(f"Loading validation data: {val_path}")

    # Confirm strict input-isolation: only train and val are loaded
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Validate feature presence and ordering
    for feat in EXPECTED_FEATURES:
        if feat not in df_train.columns:
            raise KeyError(f"Missing required feature '{feat}' in training data: {train_path}")
        if feat not in df_val.columns:
            raise KeyError(f"Missing required feature '{feat}' in validation data: {val_path}")

    if "label" not in df_train.columns:
        raise KeyError(f"Missing 'label' column in training data: {train_path}")
    if "label" not in df_val.columns:
        raise KeyError(f"Missing 'label' column in validation data: {val_path}")

    # Extract exactly the 30 model features (guaranteeing sample_id is never used)
    X_train = df_train[EXPECTED_FEATURES].values.astype(np.float64)
    y_train = df_train["label"].values.astype(np.int64)
    X_val = df_val[EXPECTED_FEATURES].values.astype(np.float64)
    y_val = df_val["label"].values.astype(np.int64)

    # Validate labels strictly binary {0, 1}
    train_labels = set(np.unique(y_train))
    val_labels = set(np.unique(y_val))
    if train_labels != {0, 1}:
        raise ValueError(f"Training labels must be exactly {{0, 1}}, found: {train_labels}")
    if val_labels != {0, 1}:
        raise ValueError(f"Validation labels must be exactly {{0, 1}}, found: {val_labels}")

    print(f"Train samples: {len(X_train)} (Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)})")
    print(f"Val samples:   {len(X_val)} (Class 0: {np.sum(y_val == 0)}, Class 1: {np.sum(y_val == 1)})")
    print(f"Feature count: {X_train.shape[1]} features (all 30 verified)")
    print("------------------------------------------------------------------")

    # Fit StandardScaler STRICTLY on training partition
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    results: List[Dict[str, Any]] = []

    for cw in class_weights:
        for c in c_values:
            clf = LogisticRegression(
                C=c,
                class_weight=cw,
                max_iter=max_iter,
                random_state=random_state,
                solver=solver,
            )
            clf.fit(X_train_scaled, y_train)

            train_pred = clf.predict(X_train_scaled)
            val_pred = clf.predict(X_val_scaled)

            # Metric evaluation
            train_macro_f1 = float(f1_score(y_train, train_pred, average="macro", zero_division=0))
            val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
            val_acc = float(accuracy_score(y_val, val_pred))

            # Per-class metrics
            prec_0 = float(precision_score(y_val, val_pred, pos_label=0, zero_division=0))
            prec_1 = float(precision_score(y_val, val_pred, pos_label=1, zero_division=0))
            rec_0 = float(recall_score(y_val, val_pred, pos_label=0, zero_division=0))
            rec_1 = float(recall_score(y_val, val_pred, pos_label=1, zero_division=0))
            f1_0 = float(f1_score(y_val, val_pred, pos_label=0, zero_division=0))
            f1_1 = float(f1_score(y_val, val_pred, pos_label=1, zero_division=0))

            # Explicit confusion matrix with canonical label order [0, 1]
            cm = confusion_matrix(y_val, val_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            n_iter = int(clf.n_iter_[0])
            converged = n_iter < max_iter
            conv_status = f"Converged ({n_iter} iters)" if converged else f"Reached max_iter ({n_iter})"

            row = {
                "C": c,
                "class_weight": str(cw),
                "solver": solver,
                "random_state": random_state,
                "n_iter": n_iter,
                "convergence_status": conv_status,
                "train_macro_f1": train_macro_f1,
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "prec_normal_0": prec_0,
                "rec_normal_0": rec_0,
                "f1_normal_0": f1_0,
                "prec_arrhythmia_1": prec_1,
                "rec_arrhythmia_1": rec_1,
                "f1_arrhythmia_1": f1_1,
                "confusion_matrix": cm.tolist(),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            results.append(row)

    df_results = pd.DataFrame(results)

    # Model Selection: Primary sorting by validation macro-F1 (descending).
    # Tie-breaking logic: In case of equal validation macro-F1 across all models,
    # prioritize canonical regularisation (C=1.0, class_weight=None).
    tied = df_results[df_results["val_macro_f1"] == df_results["val_macro_f1"].max()]
    default_candidate = tied[(tied["C"] == 1.0) & (tied["class_weight"] == "None")]
    if not default_candidate.empty:
        best_idx = default_candidate.index[0]
    else:
        best_idx = int(df_results["val_macro_f1"].idxmax())

    best_config = results[best_idx]

    return df_results, best_config


def print_summary_table(df_results: pd.DataFrame, best_config: Dict[str, Any]) -> None:
    """Print clean terminal comparison table."""
    cols_to_print = [
        "C",
        "class_weight",
        "convergence_status",
        "train_macro_f1",
        "val_macro_f1",
        "val_accuracy",
        "f1_normal_0",
        "f1_arrhythmia_1",
    ]
    print("\nHyperparameter Evaluation Table:")
    print(df_results[cols_to_print].to_string(index=False))

    print("\n------------------------------------------------------------------")
    print("BEST CONFIGURATION SELECTED (Validation Macro-F1):")
    print(f"  C:                   {best_config['C']}")
    print(f"  class_weight:        {best_config['class_weight']}")
    print(f"  solver:              {best_config['solver']}")
    print(f"  random_state:        {best_config['random_state']}")
    print(f"  convergence_status:  {best_config['convergence_status']}")
    print(f"  Train Macro-F1:      {best_config['train_macro_f1']:.6f}")
    print(f"  Val Macro-F1:        {best_config['val_macro_f1']:.6f}")
    print(f"  Val Accuracy:        {best_config['val_accuracy']:.6f}")
    print(f"  Confusion Matrix:    TN={best_config['tn']}, FP={best_config['fp']}, FN={best_config['fn']}, TP={best_config['tp']}")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Train Leakage-safe Classical Logistic Regression Baseline for PS-Q1")
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    args = parser.parse_args()

    project_root = resolve_project_root(args.root)

    train_path = project_root / "data" / "q1" / "ecg_train.csv"
    val_path = project_root / "data" / "q1" / "ecg_val.csv"

    if not train_path.exists() or not val_path.exists():
        print(f"Error: dataset files missing at {train_path} or {val_path}", file=sys.stderr)
        sys.exit(1)

    df_results, best_config = run_baseline_grid(train_path, val_path)
    print_summary_table(df_results, best_config)

    # Save results table for reporting with explicit UTF-8 encoding
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_csv = reports_dir / "classical_baseline_metrics.csv"
    df_results.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved full grid metrics to {output_csv}")


if __name__ == "__main__":
    main()
