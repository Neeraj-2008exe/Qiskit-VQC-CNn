"""
qff_qml_benchmark.py - official Task A / Task B measurement code, QFF2026 QML Track (kit v3).

Every team measures Task A and Task B with these functions, so results are
comparable. You may read and copy the code. Do not change the settings.

    from qff_qml_benchmark import task_a_bell, load_processor_d, validate_processor_d, task_b_qft3

Command line (validates your file and prints the Task B numbers to paste into it):
    python benchmark/qff_qml_benchmark.py processor_d_config.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit.transpiler import CouplingMap, StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "noise_model"))
from benchmark_noise_model import (DEFAULT_BASIS, NOISY_1Q_GATES, SEED, VERSION,  # noqa: E402
                                   assert_all_gates_noisy, get_benchmark_noise_model,
                                   prepare_for_noisy_run)

TASK_A_SHOTS = 8192
ALLOWED_NATIVE_1Q = list(NOISY_1Q_GATES)
ALLOWED_NATIVE_2Q = ["cx", "cz"]          # swap is not allowed as a native gate
D_MIN_QUBITS, D_MAX_QUBITS = 3, 9
_PLACEHOLDER_RESULTS = {"qft3_circuit_depth_before_transpile": 0, "qft3_circuit_depth_after_transpile": 0,
                        "qft3_two_qubit_gate_count": 0, "qft3_swap_overhead": 0,
                        "qft3_fidelity_ideal": 0.0, "qft3_fidelity_noisy": 0.0,
                        "benchmark_noise_model_version": VERSION}


# ---------------------------------------------------------------------------
# Task A - Bell state |Phi+> = (|00> + |11>)/sqrt(2)
# ---------------------------------------------------------------------------
def _bell(basis: str) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    if basis == "XX":
        qc.h([0, 1])
    elif basis == "YY":
        qc.sdg([0, 1])
        qc.h([0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


def _parity(counts: dict) -> tuple[float, float]:
    n = sum(counts.values())
    even = sum(c for k, c in counts.items() if k.count("1") % 2 == 0)
    e = (2 * even - n) / n
    return e, float(np.sqrt(max(1.0 - e * e, 0.0) / n))


def task_a_bell(shots: int = TASK_A_SHOTS) -> dict:
    """Official Task A numbers, ideal and noisy.

    Fidelity definition (exact for |Phi+>, because |Phi+><Phi+| = (II + XX - YY + ZZ)/4):
        F = (1 + <XX> - <YY> + <ZZ>) / 4,   SE(F) = sqrt(SE_XX^2 + SE_YY^2 + SE_ZZ^2) / 4
    <P> = (N_even - N_odd)/N with SE = sqrt((1 - <P>^2)/N).
    Note: P(00)+P(11) is a population overlap, NOT the state fidelity.
    """
    out = {"shots": shots, "noise_model_version": VERSION}
    ideal_sim = AerSimulator(seed_simulator=SEED)
    noisy_sim = AerSimulator(noise_model=get_benchmark_noise_model(), seed_simulator=SEED)
    for cond, sim in (("ideal", ideal_sim), ("noisy", noisy_sim)):
        res, var = {}, 0.0
        for b in ("XX", "YY", "ZZ"):
            tqc = prepare_for_noisy_run(_bell(b))           # rx, ry, rz, cx only
            e, se = _parity(sim.run(tqc, shots=shots).result().get_counts())
            res[b], res[b + "_se"] = e, se
            var += se ** 2
        res["fidelity"] = (1 + res["XX"] - res["YY"] + res["ZZ"]) / 4
        res["fidelity_se"] = float(np.sqrt(var) / 4)
        out[cond] = res
    return out


# ---------------------------------------------------------------------------
# Task B - Processor D
# ---------------------------------------------------------------------------
def load_processor_d(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _coupling(cfg: dict) -> CouplingMap:
    cm = CouplingMap([[int(a), int(b)] for a, b in cfg["coupling_map"]])  # DIRECTED, as written
    for q in range(cfg["n_qubits"]):
        if q not in cm.physical_qubits:
            cm.add_physical_qubit(q)
    return cm


def validate_processor_d(cfg: dict, schema_path: str | None = None,
                         require_results: bool = True) -> list[str]:
    """Returns a list of problems (empty = valid). Checks the JSON schema and the design rules.
    require_results=False skips the benchmark_results check (use it before you have run Task B)."""
    errs = []
    if not require_results:
        cfg = dict(cfg)
        cfg["benchmark_results"] = _PLACEHOLDER_RESULTS
    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "schemas", "processor_d_schema.json")
    try:
        import jsonschema
        with open(schema_path) as f:
            schema = json.load(f)
        v = jsonschema.Draft7Validator(schema)
        errs += [f"schema: {'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
                 for e in v.iter_errors(cfg)]
    except ImportError:
        errs.append("jsonschema not installed (pip install -r requirements.txt)")
    if errs:
        return errs
    n = cfg["n_qubits"]
    seen = set()
    und = {q: set() for q in range(n)}
    for a, b in cfg["coupling_map"]:
        if not (0 <= a < n and 0 <= b < n) or a == b:
            errs.append(f"coupling_map: invalid pair {[a, b]}")
            continue
        if (a, b) in seen:
            errs.append(f"coupling_map: duplicate pair {[a, b]}")
        seen.add((a, b))
        und[a].add(b)
        und[b].add(a)
    reach, dq = {0}, deque([0])
    while dq:
        for nb in und[dq.popleft()]:
            if nb not in reach:
                reach.add(nb)
                dq.append(nb)
    if len(reach) != n:
        errs.append("coupling_map: every qubit must be connected (ignoring direction)")
    gates = cfg["native_gate_set"]
    if not any(g in ALLOWED_NATIVE_2Q for g in gates):
        errs.append("native_gate_set: needs one two-qubit gate (cx or cz)")
    if not errs:
        try:
            transpile(_qft3_logical(), coupling_map=_coupling(cfg), basis_gates=gates,
                      optimization_level=1, seed_transpiler=SEED)
        except Exception as e:  # noqa: BLE001
            errs.append(f"native_gate_set is not universal enough to compile QFT-3: {e}")
    return errs


def _qft3_logical() -> QuantumCircuit:
    """Official QFT-3: textbook QFT with final qubit reversal (H, controlled-phase, SWAP)."""
    qc = QuantumCircuit(3, name="QFT3")
    for j in reversed(range(3)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(np.pi / 2 ** (j - k), k, j)
    qc.swap(0, 2)
    return qc


def _routing_swaps(qc: QuantumCircuit, cm: CouplingMap, basis) -> int:
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=basis,
                                      coupling_map=cm, seed_transpiler=SEED)
    routed = StagedPassManager(stages=["init", "layout", "routing"], init=pm.init,
                               layout=pm.layout, routing=pm.routing).run(qc)
    return routed.count_ops().get("swap", 0)


def task_b_qft3(cfg: dict) -> dict:
    """Official Task B benchmark for a Processor D config.

    Transpilation: directed coupling map as written, basis = your native_gate_set,
    optimization_level=1, seed_transpiler=42 (one-way edges cost extra gates).

    Fidelity definition: for each of the 8 computational-basis inputs |j>, the
    ideal output is QFT|j>. The input is loaded noise-free, the transpiled
    circuit runs as a density-matrix simulation under the benchmark noise model
    (gate errors only; no measurement, so no readout error), and
        F_j = <psi_j| rho_j |psi_j>     (rho_j = reduced state of the 3 logical qubits)
        qft3_fidelity = mean of F_j over the 8 inputs.
    'ideal' is the same calculation without noise and must equal 1.0.
    """
    logical = _qft3_logical()
    cm = _coupling(cfg)
    basis = list(cfg["native_gate_set"])
    tqc = transpile(logical, coupling_map=cm, basis_gates=basis,
                    optimization_level=1, seed_transpiler=SEED)
    assert_all_gates_noisy(tqc)
    n = tqc.num_qubits
    init = tqc.layout.initial_index_layout(filter_ancillas=True)
    final = tqc.layout.final_index_layout(filter_ancillas=True)
    sims = {"ideal": AerSimulator(method="density_matrix"),
            "noisy": AerSimulator(method="density_matrix", noise_model=get_benchmark_noise_model())}
    fids = {"ideal": [], "noisy": []}
    for j in range(8):
        bits = [(j >> i) & 1 for i in range(3)]          # logical qubit i holds bit i of j
        label = ["0"] * n
        for i, bit in enumerate(bits):
            label[n - 1 - init[i]] = str(bit)
        psi = Statevector.from_label("".join(str(b) for b in reversed(bits))).evolve(logical)
        circ = QuantumCircuit(n)
        circ.set_density_matrix(DensityMatrix.from_label("".join(label)))
        circ.compose(tqc, inplace=True)
        circ.save_density_matrix(qubits=list(final), label="rho")
        for cond, sim in sims.items():
            rho = sim.run(circ).result().data()["rho"]
            fids[cond].append(float(state_fidelity(psi, DensityMatrix(rho))))
    logical_swaps = logical.count_ops().get("swap", 0)
    return {
        "qft3_circuit_depth_before_transpile": logical.depth(),
        "qft3_circuit_depth_after_transpile": tqc.depth(),
        "qft3_two_qubit_gate_count": sum(1 for i in tqc.data if i.operation.num_qubits == 2),
        "qft3_swap_overhead": max(_routing_swaps(logical, cm, basis) - logical_swaps, 0),
        "qft3_fidelity_ideal": round(float(np.mean(fids["ideal"])), 6),
        "qft3_fidelity_noisy": round(float(np.mean(fids["noisy"])), 6),
        "benchmark_noise_model_version": VERSION,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    cfg = load_processor_d(sys.argv[1])
    problems = validate_processor_d(cfg, require_results=False)
    if problems:
        print("INVALID Processor D:")
        for p in problems:
            print("  -", p)
        sys.exit(2)
    res = task_b_qft3(cfg)
    print("Processor D design is valid. Official QFT-3 benchmark_results:")
    print(json.dumps(res, indent=2))
    if cfg.get("benchmark_results") and any(cfg["benchmark_results"].get(k) != v for k, v in res.items()):
        print("WARNING: benchmark_results in your file differ from the official values above.")
