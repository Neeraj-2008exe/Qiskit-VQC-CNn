"""PS-Q1 dataset and VQC construction helpers.

Provides functions to generate the synthetic classification dataset,
preprocess and split it deterministically, construct the team-authored
4-qubit feature map and ansatz, and train the VQC using statevector simulation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def generate_dataset(
    n_samples: int = 200,
    n_features: int = 4,
    n_informative: int = 3,
    n_redundant: int = 1,
    random_state: int = 42,
    save_path: Optional[str] = "data/raw/dataset.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the synthetic binary dataset for PS-Q1.

    Args:
        n_samples: Total number of samples (200).
        n_features: Total number of features (4).
        n_informative: Number of informative features (3).
        n_redundant: Number of redundant features (1).
        random_state: Deterministic random seed (42).
        save_path: Optional CSV output path to save raw dataset.

    Returns:
        Tuple of (X, y) feature array and binary label array.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y
        df.to_csv(save_path, index=False)
    return X, y


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
    feature_range: Tuple[float, float] = (0.0, np.pi),
    save_dir: Optional[str] = "data/processed",
) -> Dict[str, np.ndarray]:
    """Split and scale the dataset for quantum encoding and classical baseline.

    Features are scaled into [0, pi] using MinMaxScaler fitted on train data.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target label vector of shape (n_samples,).
        test_size: Proportion of dataset for test set (0.20 -> 40 samples).
        random_state: Deterministic seed (42).
        feature_range: Target range for min-max scaling.
        save_dir: Optional directory to save train/test CSVs.

    Returns:
        Dictionary containing X_train, X_test, y_train, y_test,
        and unscaled versions.
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler(feature_range=feature_range)
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X.shape[1])])
        train_df["target"] = y_train
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)

        test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X.shape[1])])
        test_df["target"] = y_test
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "scaler": scaler,
    }


def build_feature_map(n_qubits: int = 4, name: str = "FeatureMap") -> Tuple[QuantumCircuit, ParameterVector]:
    """Build the team-authored 4-qubit quantum feature map.

    Structure:
    - Layer 1: Hadamard gates on all qubits to create equal superposition.
    - Layer 2: Rz rotation parameterized by features x_i.
    - Layer 3: Circular CX entanglement (0->1, 1->2, 2->3, 3->0).
    - Layer 4: Second Rz rotation parameterized by x_i.

    Args:
        n_qubits: Number of qubits (4).
        name: Name for the circuit block.

    Returns:
        Tuple of (feature_map_circuit, x_parameters).
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name=name)
    for i in range(n_qubits):
        qc.h(i)
        qc.rz(x[i], i)
    for i in range(n_qubits):
        qc.cx(i, (i + 1) % n_qubits)
    for i in range(n_qubits):
        qc.rz(x[i], i)
    return qc, x


def build_ansatz(
    n_qubits: int = 4,
    reps: int = 2,
    name: str = "Ansatz",
) -> Tuple[QuantumCircuit, ParameterVector]:
    """Build the team-authored 4-qubit variational ansatz.

    Structure:
    - `reps` layers of Ry(theta) rotations followed by circular CX entanglement.
    - Final layer of Ry(theta) rotations.
    - Total parameters: (reps + 1) * n_qubits = 12 parameters for reps=2.

    Args:
        n_qubits: Number of qubits (4).
        reps: Number of variational layers (2).
        name: Name for the circuit block.

    Returns:
        Tuple of (ansatz_circuit, theta_parameters).
    """
    n_params = (reps + 1) * n_qubits
    theta = ParameterVector("theta", n_params)
    qc = QuantumCircuit(n_qubits, name=name)

    param_idx = 0
    for _ in range(reps):
        for i in range(n_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)

    for i in range(n_qubits):
        qc.ry(theta[param_idx], i)
        param_idx += 1

    return qc, theta


def build_vqc_circuit(
    feature_map: QuantumCircuit,
    ansatz: QuantumCircuit,
    measure: bool = False,
    name: str = "VQC_Logical",
) -> QuantumCircuit:
    """Compose the feature map and ansatz into a single logical VQC circuit.

    Args:
        feature_map: Feature map QuantumCircuit.
        ansatz: Variational ansatz QuantumCircuit.
        measure: If True, appends measurements for all qubits.
        name: Circuit name.

    Returns:
        Composed QuantumCircuit.
    """
    n_qubits = feature_map.num_qubits
    if measure:
        qc = QuantumCircuit(n_qubits, n_qubits, name=name)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        qc.measure(range(n_qubits), range(n_qubits))
    else:
        qc = QuantumCircuit(n_qubits, name=name)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
    return qc


def train_vqc(
    vqc_circuit: QuantumCircuit,
    x_params: ParameterVector,
    theta_params: ParameterVector,
    X_train: np.ndarray,
    y_train: np.ndarray,
    maxiter: int = 80,
    random_state: int = 42,
    observable: Optional[SparsePauliOp] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Train the VQC using statevector simulation and COBYLA optimizer.

    Pre-binds feature values for all training samples to accelerate optimization.

    Args:
        vqc_circuit: Unmeasured VQC QuantumCircuit containing x and theta parameters.
        x_params: ParameterVector for data inputs.
        theta_params: ParameterVector for trainable weights.
        X_train: Scaled training features.
        y_train: Training labels (0 or 1).
        maxiter: Maximum COBYLA iterations.
        random_state: Seed for initial parameter initialization.
        observable: Target observable (default: ZZZZ parity operator).

    Returns:
        Tuple of (optimal_theta_vector, loss_history_list).
    """
    if observable is None:
        observable = SparsePauliOp.from_list([("ZZZZ", 1.0)])

    # Map labels {0, 1} to {-1, +1} for parity expectation
    y_train_pm = 2.0 * y_train - 1.0

    # Pre-bind input features for training instances
    train_circuits_x = []
    for x_val in X_train:
        bind = {x_params[i]: x_val[i] for i in range(len(x_params))}
        train_circuits_x.append(vqc_circuit.assign_parameters(bind))

    loss_history: List[float] = []

    def cost_fn(w: np.ndarray) -> float:
        bind_theta = {theta_params[i]: w[i] for i in range(len(w))}
        preds = []
        for c in train_circuits_x:
            assigned = c.assign_parameters(bind_theta)
            sv = Statevector.from_instruction(assigned)
            preds.append(sv.expectation_value(observable).real)
        loss = float(np.mean((np.array(preds) - y_train_pm) ** 2))
        loss_history.append(loss)
        return loss

    rng = np.random.default_rng(random_state)
    init_theta = rng.uniform(-np.pi, np.pi, len(theta_params))

    # Evaluate initial loss
    cost_fn(init_theta)

    res = minimize(
        cost_fn,
        init_theta,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False},
    )

    return res.x, loss_history


def predict_vqc_ideal(
    vqc_circuit: QuantumCircuit,
    x_params: ParameterVector,
    theta_params: ParameterVector,
    theta_values: np.ndarray,
    X: np.ndarray,
    observable: Optional[SparsePauliOp] = None,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ideal statevector expectation values and binary predictions.

    Args:
        vqc_circuit: Unmeasured VQC QuantumCircuit.
        x_params: Data ParameterVector.
        theta_params: Variational ParameterVector.
        theta_values: Optimized variational weights.
        X: Feature matrix.
        observable: Target observable (default: ZZZZ).
        threshold: Decision boundary on expectation value (default: 0.0).

    Returns:
        Tuple of (continuous_expectation_values, binary_predictions).
    """
    if observable is None:
        observable = SparsePauliOp.from_list([("ZZZZ", 1.0)])

    bind_theta = {theta_params[i]: theta_values[i] for i in range(len(theta_values))}
    qc_theta = vqc_circuit.assign_parameters(bind_theta)

    expectations = []
    for x_val in X:
        bind_x = {x_params[i]: x_val[i] for i in range(len(x_params))}
        c = qc_theta.assign_parameters(bind_x)
        sv = Statevector.from_instruction(c)
        expectations.append(sv.expectation_value(observable).real)

    exp_arr = np.array(expectations)
    binary_preds = (exp_arr >= threshold).astype(int)
    return exp_arr, binary_preds
