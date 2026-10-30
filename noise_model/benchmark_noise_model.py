"""
QFF2026 QML Track - Organizer Benchmark Noise Model  (version QFF2026-v2)

Use this file for ALL noisy simulations: Task A, your PS model and the Task B
Processor D benchmark. Do not modify the error rates.

What changed from QFF2026-v1
----------------------------
v1 attached errors only to rx, ry, rz, h, x, y, z, s, t, cx, cz, swap.
Gates such as sx, sdg, tdg, u, p, cp or rzz therefore ran WITHOUT noise.
v2 fixes this in two ways:
  1. More gates carry noise (sx, sdg, tdg added).
  2. Every noisy run must first be transpiled to a basis whose gates are all
     covered. Use `prepare_for_noisy_run()` or `run_noisy()`, which do this
     and refuse any circuit that still contains an un-noised gate.
"""
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

VERSION = "QFF2026-v2"
P1 = 0.01          # single-qubit gate depolarizing error
P2 = 0.05          # two-qubit gate depolarizing error
P_READOUT = 0.02   # symmetric measurement bit-flip probability

NOISY_1Q_GATES = ["rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"]
NOISY_2Q_GATES = ["cx", "cz", "swap"]
NON_GATE_OPS = {"measure", "barrier", "reset"}

# Official basis for PS-model noisy runs and Task A (all gates carry noise).
DEFAULT_BASIS = ["rx", "ry", "rz", "cx"]
SEED = 42


def get_benchmark_noise_model():
    """Organizer-defined benchmark NoiseModel for Qiskit-Aer (identical for all teams)."""
    nm = NoiseModel(basis_gates=NOISY_1Q_GATES + NOISY_2Q_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(P1, 1), NOISY_1Q_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(P2, 2), NOISY_2Q_GATES)
    nm.add_all_qubit_readout_error(ReadoutError([[1 - P_READOUT, P_READOUT],
                                                 [P_READOUT, 1 - P_READOUT]]))
    return nm


def get_noise_params():
    return {"single_qubit_depolarizing": P1,
            "two_qubit_depolarizing": P2,
            "readout_error": P_READOUT,
            "noisy_1q_gates": NOISY_1Q_GATES,
            "noisy_2q_gates": NOISY_2Q_GATES,
            "default_basis": DEFAULT_BASIS,
            "simulator": "qiskit-aer AerSimulator",
            "version": VERSION}


def unnoised_gates(circuit):
    """Names of operations in `circuit` that the benchmark noise model would NOT add noise to."""
    covered = set(NOISY_1Q_GATES) | set(NOISY_2Q_GATES) | NON_GATE_OPS
    return sorted({inst.operation.name for inst in circuit.data} - covered)


def assert_all_gates_noisy(circuit):
    bad = unnoised_gates(circuit)
    if bad:
        raise ValueError(f"Circuit contains gates with no benchmark noise: {bad}. "
                         f"Transpile with prepare_for_noisy_run() first.")


def prepare_for_noisy_run(circuit, basis_gates=None, coupling_map=None):
    """Transpile to a fully-noised basis (default rx, ry, rz, cx), opt level 1, seed 42."""
    basis = list(basis_gates) if basis_gates else DEFAULT_BASIS
    tqc = transpile(circuit, basis_gates=basis, coupling_map=coupling_map,
                    optimization_level=1, seed_transpiler=SEED)
    assert_all_gates_noisy(tqc)
    return tqc


def noisy_simulator(seed=SEED, method="automatic"):
    return AerSimulator(noise_model=get_benchmark_noise_model(), seed_simulator=seed,
                        method=method)


def run_noisy(circuit, shots=8192, seed=SEED, basis_gates=None, coupling_map=None):
    """Transpile + run under the benchmark noise model. Returns counts."""
    tqc = prepare_for_noisy_run(circuit, basis_gates, coupling_map)
    return noisy_simulator(seed).run(tqc, shots=shots).result().get_counts()


def pennylane_noisy_device(wires, shots=8192, seed=SEED):
    """PennyLane device that runs on Qiskit-Aer with the benchmark noise model.
    Needs the pinned `pennylane-qiskit` plugin. Circuits are transpiled to the
    fully-noised DEFAULT_BASIS before they run, so no gate escapes the noise."""
    import pennylane as qml
    return qml.device("qiskit.aer", wires=wires, shots=shots,
                      noise_model=get_benchmark_noise_model(),
                      seed_simulator=seed, seed_transpiler=SEED,
                      basis_gates=DEFAULT_BASIS, optimization_level=1)


if __name__ == "__main__":
    get_benchmark_noise_model()
    print("Benchmark noise model created successfully.")
    print("Parameters:", get_noise_params())
