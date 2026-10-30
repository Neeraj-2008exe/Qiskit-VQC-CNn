"""Processor configuration and noise-model loading helpers.

Provides functions to parse participant-defined processor configurations
(Processor A and Processor B) and construct reproducible Qiskit Aer noise models.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def load_processor_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a processor configuration JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Dictionary containing processor parameters: name, num_qubits,
        basis_gates, coupling_map, noise, transpile, shots, and note.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = ["name", "num_qubits", "basis_gates", "coupling_map", "noise", "transpile", "shots"]
    for k in required_keys:
        if k not in cfg:
            raise KeyError(f"Configuration file {config_path} missing required key: {k}")

    return cfg


def build_noise_model(config: Dict[str, Any]) -> NoiseModel:
    """Build a Qiskit Aer NoiseModel from the processor configuration.

    Applies depolarizing error to single-qubit basis gates ('sx', 'x'),
    depolarizing error to two-qubit basis gates ('cx'), and readout error
    to measurements.

    Args:
        config: Processor configuration dictionary.

    Returns:
        Configured NoiseModel instance.
    """
    basis_gates = config.get("basis_gates", ["rz", "sx", "x", "cx"])
    noise_cfg = config.get("noise", {})

    p_1q = noise_cfg.get("single_qubit_depolarizing", 0.001)
    p_2q = noise_cfg.get("cx_depolarizing", 0.005)
    p_ro = noise_cfg.get("readout_error", 0.01)

    nm = NoiseModel(basis_gates=basis_gates)

    if p_1q > 0.0:
        err_1q = depolarizing_error(p_1q, 1)
        # Apply to single-qubit microwave drive gates
        nm.add_all_qubit_quantum_error(err_1q, ["sx", "x"])

    if p_2q > 0.0:
        err_2q = depolarizing_error(p_2q, 2)
        nm.add_all_qubit_quantum_error(err_2q, ["cx"])

    if p_ro > 0.0:
        ro_err = ReadoutError([[1.0 - p_ro, p_ro], [p_ro, 1.0 - p_ro]])
        nm.add_all_qubit_readout_error(ro_err)

    return nm


def get_coupling_map(config: Dict[str, Any]) -> CouplingMap:
    """Extract and instantiate CouplingMap from processor configuration.

    Args:
        config: Processor configuration dictionary.

    Returns:
        CouplingMap instance.
    """
    return CouplingMap(config["coupling_map"])


def get_simulator(config: Dict[str, Any], noisy: bool = True) -> AerSimulator:
    """Instantiate an AerSimulator configured for the processor.

    Args:
        config: Processor configuration dictionary.
        noisy: If True, attaches the processor noise model.

    Returns:
        AerSimulator instance.
    """
    if noisy:
        noise_model = build_noise_model(config)
        return AerSimulator(noise_model=noise_model)
    return AerSimulator()
