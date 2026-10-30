"""Transpilation and layout/routing helpers.

Provides functions to transpile quantum circuits onto target hardware topologies,
extract gate counts, circuit depths, SWAP/routing overhead, and initial qubit layouts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap


def transpile_for_processor(circuit: QuantumCircuit, config: Dict[str, Any]) -> QuantumCircuit:
    """Transpile a logical circuit onto the specified target processor.

    Args:
        circuit: Logical QuantumCircuit to transpile.
        config: Processor configuration dictionary containing coupling_map,
            basis_gates, and transpile settings.

    Returns:
        Transpiled physical QuantumCircuit.
    """
    cmap = CouplingMap(config["coupling_map"])
    transpile_cfg = config.get("transpile", {})
    opt_level = transpile_cfg.get("optimization_level", 1)
    seed = transpile_cfg.get("seed_transpiler", 42)

    return transpile(
        circuit,
        coupling_map=cmap,
        basis_gates=config["basis_gates"],
        optimization_level=opt_level,
        seed_transpiler=seed,
    )


def extract_circuit_metrics(
    circuit: QuantumCircuit,
    name: str = "Circuit",
    logical_circuit: Optional[QuantumCircuit] = None,
) -> Dict[str, Any]:
    """Extract structural metrics, gate breakdown, and routing overhead from a circuit.

    Args:
        circuit: The target QuantumCircuit (logical or physical).
        name: Identifier name for the circuit/processor.
        logical_circuit: Optional baseline logical circuit to calculate routing overhead.

    Returns:
        Dictionary of metrics including depth, gate counts, CX overhead,
        and estimated SWAP count.
    """
    ops = circuit.count_ops()
    cx_count = int(ops.get("cx", 0))
    rz_count = int(ops.get("rz", 0))
    sx_count = int(ops.get("sx", 0))
    x_count = int(ops.get("x", 0))
    measure_count = int(ops.get("measure", 0))
    depth = circuit.depth()
    total_ops = circuit.size()

    metrics: Dict[str, Any] = {
        "Name": name,
        "Total Qubits": circuit.num_qubits,
        "Depth": depth,
        "Total Operations": total_ops,
        "CX Count": cx_count,
        "RZ Count": rz_count,
        "SX Count": sx_count,
        "X Count": x_count,
        "Measure Count": measure_count,
    }

    if logical_circuit is not None:
        logical_ops = logical_circuit.count_ops()
        logical_cx = int(logical_ops.get("cx", 0))
        logical_depth = logical_circuit.depth()

        extra_cx = max(0, cx_count - logical_cx)
        cx_ratio = cx_count / max(1, logical_cx)
        # Each SWAP gate decomposed into standard basis requires 3 CX gates
        estimated_swaps = extra_cx // 3

        metrics["Logical CX"] = logical_cx
        metrics["Extra CX (Routing)"] = extra_cx
        metrics["CX Overhead Ratio"] = round(cx_ratio, 3)
        metrics["Estimated SWAPs"] = estimated_swaps
        metrics["Depth Overhead"] = depth - logical_depth
    else:
        metrics["Logical CX"] = cx_count
        metrics["Extra CX (Routing)"] = 0
        metrics["CX Overhead Ratio"] = 1.0
        metrics["Estimated SWAPs"] = 0
        metrics["Depth Overhead"] = 0

    return metrics


def build_transpilation_summary(
    logical_circuit: QuantumCircuit,
    transpiled_A: QuantumCircuit,
    transpiled_B: QuantumCircuit,
) -> pd.DataFrame:
    """Build a comprehensive DataFrame comparing logical, Processor A, and Processor B circuits.

    Args:
        logical_circuit: Untranspiled logical circuit.
        transpiled_A: Physical circuit transpiled for Processor A.
        transpiled_B: Physical circuit transpiled for Processor B.

    Returns:
        Pandas DataFrame with transpilation metrics.
    """
    m_log = extract_circuit_metrics(logical_circuit, name="Logical (Ideal)")
    m_a = extract_circuit_metrics(transpiled_A, name="Processor A (5Q Linear)", logical_circuit=logical_circuit)
    m_b = extract_circuit_metrics(transpiled_B, name="Processor B (9Q Mesh)", logical_circuit=logical_circuit)

    df = pd.DataFrame([m_log, m_a, m_b])
    return df
