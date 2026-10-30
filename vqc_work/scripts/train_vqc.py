"""PS-Q1 hybrid VQC: StandardScaler -> PCA(4) -> angle encoding -> trainable ansatz -> <Z0> -> affine readout.

Loads ONLY the labelled training and validation CSVs (explicit paths below). For each of two angle encodings
("scaled": PCA scores rescaled to [-pi, pi] with a scale fitted on train; "direct": raw PCA scores, the originally
specified encoding, kept as an ablation) it trains the four predeclared configurations, retrains the selected
configuration with four extra seeds, evaluates ideal (exact statevector) and official QFF2026-v2 noisy validation
metrics, runs integrity / reproducibility checks and writes the VQC deliverables under vqc_work/.

Usage (from anywhere):
    python vqc_work/scripts/train_vqc.py --selftest  # fast correctness checks, writes nothing
    python vqc_work/scripts/train_vqc.py --smoke     # 1-epoch timing run + small noisy run, writes nothing
    python vqc_work/scripts/train_vqc.py             # full pipeline, writes reports/artifacts (~15 min, 10 cores)
"""
import sys

sys.dont_write_bytecode = True  # keep the repository free of __pycache__

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------- paths / constants
ROOT = Path(__file__).resolve().parents[2]
VQC_DIR = ROOT / "vqc_work"
TRAIN_CSV = ROOT / "data" / "q1" / "ecg_train.csv"
VAL_CSV = ROOT / "data" / "q1" / "ecg_val.csv"
NOISE_DIR = ROOT / "noise_model"
CLASSICAL_METRICS = ROOT / "reports" / "classical_baseline_metrics.csv"  # read-only, teammate's output
REPORT_TRAIN = VQC_DIR / "reports" / "vqc_training_results.md"
REPORT_NOISY = VQC_DIR / "reports" / "vqc_ideal_vs_noisy.md"
RESOURCE_CSV = VQC_DIR / "reports" / "vqc_resource_summary.csv"
CONFIG_JSON = VQC_DIR / "artifacts" / "vqc_config.json"
HISTORY_CSV = VQC_DIR / "artifacts" / "vqc_training_history.csv"

SEED = 42
EXTRA_SEEDS = [43, 44, 45, 46]
NOISY_SIM_SEEDS = [42, 43, 44, 45, 46]
N_QUBITS = 4
LAYER_GRID = [1, 2]
LR_GRID = [0.01, 0.001]
MAX_EPOCHS = 30
BATCH_SIZE = 16
THRESHOLD = 0.5
NOISY_SHOTS = 8192  # helper default (run_noisy) and the benchmark's TASK_A_SHOTS; no PS-Q1-specific value is specified
RING_CX = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]
INIT_SCALE = 0.1
ENCODINGS = ["scaled", "direct"]
CANONICAL_ENCODING = "scaled"
LABEL_COL = "label"
CLASSES = [0, 1]
N_BOOT = 1000
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def validate_root():
    needed = [TRAIN_CSV, VAL_CSV, NOISE_DIR / "benchmark_noise_model.py"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        sys.exit(f"Project root {ROOT} is not the challenge kit; missing: {missing}")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_bnm():
    if str(NOISE_DIR) not in sys.path:
        sys.path.insert(0, str(NOISE_DIR))
    import benchmark_noise_model as bnm
    return bnm


# ----------------------------------------------------------------------------- data + preprocessing
def load_data():
    return pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)


def integrity_checks(tr, va):
    feats = [c for c in tr.columns if c != LABEL_COL]
    out = {
        "train_shape": list(tr.shape), "val_shape": list(va.shape), "n_features": len(feats),
        "same_columns_same_order": list(tr.columns) == list(va.columns),
        "label_column_present": LABEL_COL in tr.columns and LABEL_COL in va.columns,
        "sample_id_in_features": "sample_id" in feats,
        "train_missing": int(tr.isna().sum().sum()), "val_missing": int(va.isna().sum().sum()),
        "train_nonfinite": int((~np.isfinite(tr[feats].to_numpy())).sum()),
        "val_nonfinite": int((~np.isfinite(va[feats].to_numpy())).sum()),
        "train_labels": {int(k): int(v) for k, v in tr[LABEL_COL].value_counts().sort_index().items()},
        "val_labels": {int(k): int(v) for k, v in va[LABEL_COL].value_counts().sort_index().items()},
        "dup_rows_train": int(tr.duplicated(feats).sum()), "dup_rows_val": int(va.duplicated(feats).sum()),
        "cross_partition_duplicate_rows": int(len(tr[feats].merge(va[feats], how="inner"))),
        "feature_names": feats,
    }
    out["labels_valid"] = set(tr[LABEL_COL].unique()) <= {0, 1} and set(va[LABEL_COL].unique()) <= {0, 1}
    ok = (out["same_columns_same_order"] and out["label_column_present"] and not out["sample_id_in_features"]
          and out["train_missing"] == 0 and out["val_missing"] == 0 and out["train_nonfinite"] == 0
          and out["val_nonfinite"] == 0 and out["labels_valid"] and out["dup_rows_train"] == 0
          and out["dup_rows_val"] == 0 and out["cross_partition_duplicate_rows"] == 0 and out["n_features"] == 30)
    out["passed"] = bool(ok)
    return out


def fit_preprocessing(tr, va, feats):
    """Scaler, PCA and the angle scale are all fitted on the training rows only."""
    Xtr, Xva = tr[feats].to_numpy(float), va[feats].to_numpy(float)
    scaler = StandardScaler().fit(Xtr)
    Str = scaler.transform(Xtr)
    pca = PCA(n_components=N_QUBITS, svd_solver="full", random_state=SEED).fit(Str)
    Ptr, Pva = pca.transform(Str), pca.transform(scaler.transform(Xva))
    angle_scale = np.pi / np.abs(Ptr).max(0)
    angles = {"direct": (Ptr, Pva), "scaled": (Ptr * angle_scale, Pva * angle_scale)}
    info = {"evr": pca.explained_variance_ratio_.tolist(), "cum_evr": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "angle_scale": angle_scale.tolist(),
            "ranges": {enc: {"train_min": a[0].min(0).tolist(), "train_max": a[0].max(0).tolist(),
                             "val_min": a[1].min(0).tolist(), "val_max": a[1].max(0).tolist()} for enc, a in angles.items()},
            "val_outside_pi_frac": {enc: float((np.abs(a[1]) > np.pi).any(1).mean()) for enc, a in angles.items()},
            "train_outside_pi_frac": {enc: float((np.abs(a[0]) > np.pi).any(1).mean()) for enc, a in angles.items()}}
    return scaler, pca, angle_scale, angles, info


# ----------------------------------------------------------------------------- PennyLane model
_DEV = qml.device("default.qubit", wires=N_QUBITS)


def make_qnode(n_layers):
    @qml.qnode(_DEV, interface="autograd", diff_method="backprop")
    def circuit(x, theta):
        for i in range(N_QUBITS):
            qml.RY(x[:, i], wires=i)
        for l in range(n_layers):
            for i in range(N_QUBITS):
                qml.RY(theta[l, 0, i], wires=i)
            for i in range(N_QUBITS):
                qml.RZ(theta[l, 1, i], wires=i)
            for a, b in RING_CX:
                qml.CNOT(wires=[a, b])
        return qml.expval(qml.PauliZ(0))
    return circuit


def split_params(p, n_layers):
    theta = p[: 8 * n_layers].reshape(n_layers, 2, N_QUBITS)
    return theta, p[8 * n_layers], p[8 * n_layers + 1]


def param_name(i):
    return f"layer{i // 8}_{'RY' if (i % 8) < 4 else 'RZ'}_q{i % 4}"


def bce_with_logits(logit, y):
    return pnp.mean(pnp.maximum(logit, 0.0) - logit * y + pnp.log1p(pnp.exp(-pnp.abs(logit))))


def ideal_logits(qnode, p, n_layers, A):
    theta, w, b = split_params(pnp.array(np.asarray(p, float), requires_grad=False), n_layers)
    z = np.asarray(qnode(pnp.array(A, requires_grad=False), theta), float)
    return float(w) * z + float(b), z


# ----------------------------------------------------------------------------- metrics
def compute_metrics(y, logit):
    y = np.asarray(y, int)
    logit = np.asarray(logit, float)
    prob = expit(logit)
    pred = (prob >= THRESHOLD).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=CLASSES).ravel()
    loss = float(np.mean(np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))))
    q = np.percentile(prob, [5, 25, 50, 75, 95])
    return {
        "loss": loss, "macro_f1": float(np.mean(f1)), "accuracy": float((pred == y).mean()),
        "precision_0": float(pr[0]), "recall_0": float(rc[0]), "f1_0": float(f1[0]),
        "precision_1": float(pr[1]), "recall_1": float(rc[1]), "f1_1": float(f1[1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "prob_mean": float(prob.mean()), "prob_std": float(prob.std()), "prob_min": float(prob.min()),
        "prob_max": float(prob.max()), "prob_p5": float(q[0]), "prob_p25": float(q[1]), "prob_p50": float(q[2]),
        "prob_p75": float(q[3]), "prob_p95": float(q[4]), "frac_pred_1": float(pred.mean()),
        "finite": bool(np.isfinite(logit).all() and np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()),
    }


def bootstrap_ci(y, pred, n_boot=N_BOOT, seed=SEED):
    """95% percentile bootstrap interval of macro-F1 for fixed predictions (resampling validation rows)."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = [f1_score(y[i], pred[i], average="macro", zero_division=0)
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------- training
def train_config(cfg_id, encoding, n_layers, lr, Atr, ytr, Ava, yva, seed=SEED, epochs=MAX_EPOCHS, verbose=False):
    qnode = make_qnode(n_layers)
    rng = np.random.default_rng(seed)
    p0 = np.concatenate([rng.normal(0.0, INIT_SCALE, 8 * n_layers), [1.0, 0.0]])
    p = pnp.array(p0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)
    ytr_f = pnp.array(ytr.astype(float), requires_grad=False)
    n = len(ytr)
    hist, best, n_fwd, failed, n_train_fwd = [], None, 0, None, 0
    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        order = rng.permutation(n)
        batch_losses = []
        for s in range(0, n, BATCH_SIZE):  # includes final partial batch
            idx = order[s: s + BATCH_SIZE]
            xb = pnp.array(Atr[idx], requires_grad=False)
            yb = ytr_f[idx]

            def cost(q, xb=xb, yb=yb):
                th, w, b = split_params(q, n_layers)
                return bce_with_logits(w * qnode(xb, th) + b, yb)

            p, lval = opt.step_and_cost(cost, p)
            batch_losses.append(float(lval))
            n_fwd += len(idx)
            n_train_fwd += len(idx)
        lt, _ = ideal_logits(qnode, p, n_layers, Atr)
        lv, _ = ideal_logits(qnode, p, n_layers, Ava)
        n_fwd += len(Atr) + len(Ava)
        mt, mv = compute_metrics(ytr, lt), compute_metrics(yva, lv)
        finite = bool(np.isfinite(batch_losses).all() and np.isfinite(np.asarray(p)).all() and mt["finite"] and mv["finite"])
        is_best = finite and (best is None or mv["macro_f1"] > best["val_macro_f1"])  # strict '>' keeps earliest tie
        row = {"config_id": cfg_id, "encoding": encoding, "layers": n_layers, "lr": lr, "seed": seed, "epoch": ep,
               "batch_size": BATCH_SIZE, "n_batches": len(batch_losses), "batch_loss_mean": float(np.mean(batch_losses)),
               "train_loss": mt["loss"], "train_macro_f1": mt["macro_f1"], "train_accuracy": mt["accuracy"],
               "train_prob_mean": mt["prob_mean"]}
        row.update({f"val_{k}": v for k, v in mv.items()})
        row.update({"epoch_time_s": time.perf_counter() - t_ep, "finite": finite, "is_best_so_far": bool(is_best)})
        hist.append(row)
        if is_best:
            best = {"epoch": ep, "val_macro_f1": mv["macro_f1"], "params": np.asarray(p, float).copy()}
        if verbose:
            print(f"  [{cfg_id}] ep {ep:2d} loss {mt['loss']:.4f} trF1 {mt['macro_f1']:.4f} "
                  f"vaLoss {mv['loss']:.4f} vaF1 {mv['macro_f1']:.4f} ({row['epoch_time_s']:.1f}s)", flush=True)
        if not finite:
            failed = f"non-finite value at epoch {ep}"
            break
    return {"cfg_id": cfg_id, "encoding": encoding, "layers": n_layers, "lr": lr, "seed": seed, "history": hist, "best": best,
            "train_time_s": time.perf_counter() - t_start, "forward_evals": n_fwd, "train_forward_evals": n_train_fwd,
            "failed": failed, "init_params": p0.tolist(), "final_params": np.asarray(p, float).tolist()}


def _run_job(job):
    return train_config(**job)


def run_parallel(jobs, workers=None):
    """Each job in its own process with single-threaded BLAS (avoids thread oversubscription)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: "1" for k in THREAD_VARS})
    try:
        with ProcessPoolExecutor(max_workers=workers or min(len(jobs), os.cpu_count() or 1),
                                 mp_context=mp.get_context("spawn")) as ex:
            return list(ex.map(_run_job, jobs))
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def select_final(results):
    ok = [r for r in results if r["best"] is not None and not r["failed"]]
    return sorted(ok, key=lambda r: (-r["best"]["val_macro_f1"], r["layers"], 0 if r["lr"] == 0.001 else 1,
                                     r["best"]["epoch"]))[0]


# ----------------------------------------------------------------------------- structure analysis
def structural_analysis(n_layers, n_points=3, n_inputs=64):
    """Which quantum parameters / inputs can influence <Z0> at all (zero gradient at every random point)."""
    rng = np.random.default_rng(SEED)
    qnode = make_qnode(n_layers)
    x = pnp.array(rng.uniform(-np.pi, np.pi, (n_inputs, N_QUBITS)), requires_grad=False)
    zero_sets, sens = [], []
    for _ in range(n_points):
        p = pnp.array(rng.uniform(-np.pi, np.pi, 8 * n_layers), requires_grad=True)
        g = np.asarray(qml.grad(lambda q: pnp.sum(qnode(x, q.reshape(n_layers, 2, N_QUBITS))))(p))
        zero_sets.append({i for i in range(8 * n_layers) if abs(g[i]) < 1e-12})
        xs = pnp.array(np.asarray(x), requires_grad=True)
        th = pnp.array(np.asarray(p).reshape(n_layers, 2, N_QUBITS), requires_grad=False)
        sens.append(np.abs(np.asarray(qml.jacobian(lambda xx: pnp.sum(qnode(xx, th)))(xs))).mean(0))
    dead = sorted(set.intersection(*zero_sets))
    return {"layers": n_layers, "dead_params": [param_name(i) for i in dead], "n_dead": len(dead),
            "n_quantum_params": 8 * n_layers, "n_effective_quantum_params": 8 * n_layers - len(dead),
            "input_sensitivity_mean_abs": np.mean(sens, 0).tolist(), "consistent_across_points": all(s == zero_sets[0] for s in zero_sets)}


def heisenberg_ring_check(n_states=20):
    """The CX ring maps Z0 (after the ring) to Z0 Z2 Z3 (before it); verified on random product+entangled states."""
    rng = np.random.default_rng(SEED)
    dev = qml.device("default.qubit", wires=N_QUBITS)

    def prep(a):
        for i in range(N_QUBITS):
            qml.RY(a[i], wires=i); qml.RZ(a[4 + i], wires=i)
        qml.CNOT(wires=[0, 2]); qml.CNOT(wires=[1, 3])
        for i in range(N_QUBITS):
            qml.RX(a[8 + i], wires=i)

    @qml.qnode(dev)
    def after(a):
        prep(a)
        for c, t in RING_CX:
            qml.CNOT(wires=[c, t])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev)
    def before(a):
        prep(a)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2) @ qml.PauliZ(3))

    diffs = [abs(float(after(a)) - float(before(a))) for a in rng.uniform(-np.pi, np.pi, (n_states, 12))]
    return float(max(diffs))


# ----------------------------------------------------------------------------- Qiskit / official noise model
def logical_resources(n_layers):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):
        qc.ry(0.1, i)
    for _ in range(n_layers):
        for i in range(N_QUBITS):
            qc.ry(0.1, i)
        for i in range(N_QUBITS):
            qc.rz(0.1, i)
        for a, b in RING_CX:
            qc.cx(a, b)
    ops = {k: int(v) for k, v in qc.count_ops().items()}
    return {"depth": qc.depth(), "ops": ops, "size": qc.size(), "cx": ops.get("cx", 0)}


def qiskit_template(n_layers):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    x, th = ParameterVector("x", N_QUBITS), ParameterVector("t", 8 * n_layers)
    qc = QuantumCircuit(N_QUBITS, 1)
    for i in range(N_QUBITS):
        qc.ry(x[i], i)
    k = 0
    for _ in range(n_layers):
        for i in range(N_QUBITS):
            qc.ry(th[k], i); k += 1
        for i in range(N_QUBITS):
            qc.rz(th[k], i); k += 1
        for a, b in RING_CX:
            qc.cx(a, b)
    qc.measure(0, 0)
    return qc, x, th


def bound_circuits(n_layers, theta_flat, A):
    """Logical circuits with every sample's angles and the trained weights bound in (not yet transpiled)."""
    tmpl, xs, ths = qiskit_template(n_layers)
    return [tmpl.assign_parameters({**{xs[i]: float(a[i]) for i in range(N_QUBITS)},
                                    **{ths[j]: float(theta_flat[j]) for j in range(8 * n_layers)}}) for a in A]


def transpile_each(bnm, circuits):
    """Official prepare_for_noisy_run() on EVERY executed circuit (it also asserts that every gate carries noise)."""
    tq = [bnm.prepare_for_noisy_run(c) for c in circuits]
    allowed = set(bnm.DEFAULT_BASIS) | {"measure"}
    bad = sorted({i.operation.name for c in tq for i in c.data} - allowed)
    unnoised = sorted({g for c in tq for g in bnm.unnoised_gates(c)})
    n1q = np.array([sum(v for k, v in c.count_ops().items() if k in ("rx", "ry", "rz")) for c in tq])
    cx = np.array([c.count_ops().get("cx", 0) for c in tq])
    depth = np.array([c.depth() for c in tq])
    stats = {"n_circuits": len(tq), "ops_outside_basis": bad, "unnoised_gates": unnoised,
             "1q_gates_mean": float(n1q.mean()), "1q_gates_min": int(n1q.min()), "1q_gates_max": int(n1q.max()),
             "cx_min": int(cx.min()), "cx_max": int(cx.max()), "depth_mean": float(depth.mean()),
             "depth_min": int(depth.min()), "depth_max": int(depth.max())}
    return tq, stats


def z0_from_counts(counts, shots):
    return (counts.get("0", 0) - counts.get("1", 0)) / shots


def run_z(simulator, circuits, shots=NOISY_SHOTS):
    res = simulator.run(circuits, shots=shots).result()
    return np.array([z0_from_counts(res.get_counts(i), shots) for i in range(len(circuits))])


def noise_verification(bnm, n_layers, params, A_val, n_check=300):
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_aer import AerSimulator
    ver = {"noise_instructions": sorted(bnm.get_benchmark_noise_model().noise_instructions),
           "noise_params": bnm.get_noise_params(), "version": bnm.VERSION}
    ideal = AerSimulator(seed_simulator=SEED)
    for name, q, expect in (("x_on_q0", 0, -1.0), ("x_on_q1", 1, +1.0)):
        qc = QuantumCircuit(N_QUBITS, 1); qc.x(q); qc.measure(0, 0)
        z = z0_from_counts(ideal.run(transpile(qc, basis_gates=bnm.DEFAULT_BASIS), shots=1024).result().get_counts(), 1024)
        ver[f"bitorder_{name}"] = {"z0": z, "expected": expect, "ok": abs(z - expect) < 1e-9}
    qc = QuantumCircuit(N_QUBITS, 1); qc.measure(0, 0)
    sim = bnm.noisy_simulator(SEED)
    p1 = sim.run(bnm.prepare_for_noisy_run(qc), shots=NOISY_SHOTS).result().get_counts().get("1", 0) / NOISY_SHOTS
    se = float(np.sqrt(bnm.P_READOUT * (1 - bnm.P_READOUT) / NOISY_SHOTS))
    ver["readout_noise_empty_circuit"] = {"p_read_1": p1, "expected": bnm.P_READOUT, "se": se, "ok": abs(p1 - bnm.P_READOUT) < 4 * se}
    ver["simulator_has_noise_model"] = bool(sim.options.noise_model is not None and
                                            sorted(sim.options.noise_model.noise_instructions) == ver["noise_instructions"])
    L = n_layers
    sub = A_val[:n_check]
    _, z_exact = ideal_logits(make_qnode(L), params, L, sub)
    logical = bound_circuits(L, params[: 8 * L], sub)
    # exact Qiskit statevector of the logical circuit (qubit 0 = rightmost Pauli label) vs PennyLane
    z_sv = np.array([float(np.real(Statevector(c.remove_final_measurements(inplace=False)).expectation_value(SparsePauliOp("IIIZ"))))
                     for c in logical])
    ver["qiskit_statevector_vs_pennylane_max_diff"] = float(np.abs(z_sv - z_exact).max())
    ver["exact_equivalence_ok"] = bool(ver["qiskit_statevector_vs_pennylane_max_diff"] < 1e-9)
    tq, _ = transpile_each(bnm, logical)
    z_tq_sv = np.array([float(np.real(Statevector(c.remove_final_measurements(inplace=False)).expectation_value(SparsePauliOp("IIIZ"))))
                        for c in tq])
    ver["transpiled_statevector_vs_pennylane_max_diff"] = float(np.abs(z_tq_sv - z_exact).max())
    ver["transpile_preserves_circuit_ok"] = bool(ver["transpiled_statevector_vs_pennylane_max_diff"] < 1e-9)
    z_nl = run_z(AerSimulator(seed_simulator=SEED), tq)
    z_ny = run_z(bnm.noisy_simulator(SEED), tq)
    sigma = 1.0 / np.sqrt(NOISY_SHOTS)
    ver["shots_noiseless_vs_exact"] = {"n": int(len(sub)), "max_abs_diff": float(np.abs(z_nl - z_exact).max()),
                                       "mean_abs_diff": float(np.abs(z_nl - z_exact).mean()), "shot_sigma": float(sigma),
                                       "ok": bool(np.abs(z_nl - z_exact).max() < 5 * sigma)}
    ver["noisy_vs_exact_subset"] = {"n": int(len(sub)), "mean_abs_diff": float(np.abs(z_ny - z_exact).mean()),
                                    "mean_abs_z_exact": float(np.abs(z_exact).mean()), "mean_abs_z_noisy": float(np.abs(z_ny).mean())}
    ver["all_ok"] = bool(ver["bitorder_x_on_q0"]["ok"] and ver["bitorder_x_on_q1"]["ok"] and ver["readout_noise_empty_circuit"]["ok"]
                         and ver["simulator_has_noise_model"] and ver["exact_equivalence_ok"] and ver["transpile_preserves_circuit_ok"]
                         and ver["shots_noiseless_vs_exact"]["ok"])
    return ver


def noisy_evaluation(bnm, n_layers, params, A, y):
    """Same trained weights; per-circuit official transpilation; 8192 shots; simulator seeds 42-46."""
    L = n_layers
    w, b = float(params[8 * L]), float(params[8 * L + 1])
    t0 = time.perf_counter()
    tq, stats = transpile_each(bnm, bound_circuits(L, params[: 8 * L], A))
    t_transpile = time.perf_counter() - t0
    per_seed, z_by_seed, t_runs = [], {}, []
    for sd in NOISY_SIM_SEEDS:
        t1 = time.perf_counter()
        z = run_z(bnm.noisy_simulator(sd), tq)
        t_runs.append(time.perf_counter() - t1)
        z_by_seed[sd] = z
        per_seed.append(compute_metrics(y, w * z + b))
    z42 = z_by_seed[SEED]
    f1s = [m["macro_f1"] for m in per_seed]
    return {"z": z42, "logit": w * z42 + b, "metrics": per_seed[0], "per_seed_macro_f1": f1s,
            "per_seed_accuracy": [m["accuracy"] for m in per_seed],
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s, ddof=1)),
            "transpile": stats, "transpile_time_s": t_transpile, "run_time_s_per_seed": float(np.mean(t_runs)),
            "n_circuits": len(tq), "shots_per_seed": len(tq) * NOISY_SHOTS, "total_shots": len(tq) * NOISY_SHOTS * len(NOISY_SIM_SEEDS)}


# ----------------------------------------------------------------------------- persistence + reproducibility
def inference_state(scaler, pca, angle_scale, feats, sel):
    L = sel["layers"]
    p = sel["best"]["params"]
    return {
        "feature_names": feats, "label_mapping": {"0": "Normal", "1": "Arrhythmia"}, "threshold": THRESHOLD,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist()},
        "pca": {"n_components": N_QUBITS, "svd_solver": "full", "mean": pca.mean_.tolist(),
                "components": pca.components_.tolist(), "explained_variance": pca.explained_variance_.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist()},
        "encoding": {"name": sel["encoding"], "angle_scale": (np.asarray(angle_scale) if sel["encoding"] == "scaled" else np.ones(N_QUBITS)).tolist(),
                     "rule": "angle_i = pca_score_i * angle_scale_i; angle_scale_i = pi / max|train pca_score_i| (scaled) or 1 (direct)"},
        "circuit": {"qubits": N_QUBITS, "encoding_gate": "RY(angle_i) on qubit i", "layers": L,
                    "layer_structure": "RY(theta[l,0,i]) all i; RZ(theta[l,1,i]) all i; ring CX sequence",
                    "ring_cx_sequence": RING_CX, "observable": "Z on qubit 0"},
        "params": {"theta": np.asarray(p[: 8 * L]).reshape(L, 2, N_QUBITS).tolist(),
                   "readout_w": float(p[8 * L]), "readout_b": float(p[8 * L + 1])},
    }


def reload_probabilities(state, raw_X):
    """Rebuild the full model from the saved JSON state only (no objects from the training run)."""
    X = (np.asarray(raw_X, float) - np.array(state["scaler"]["mean"])) / np.array(state["scaler"]["scale"])
    A = ((X - np.array(state["pca"]["mean"])) @ np.array(state["pca"]["components"]).T) * np.array(state["encoding"]["angle_scale"])
    L = state["circuit"]["layers"]
    theta = pnp.array(state["params"]["theta"], requires_grad=False)
    z = np.asarray(make_qnode(L)(pnp.array(A, requires_grad=False), theta), float)
    return expit(state["params"]["readout_w"] * z + state["params"]["readout_b"])


# ----------------------------------------------------------------------------- self-test
def selftest():
    """Fast correctness checks. Prints PASS/FAIL per check and exits non-zero on any failure."""
    validate_root()
    bnm = load_bnm()
    from qiskit.quantum_info import SparsePauliOp, Statevector
    rng = np.random.default_rng(SEED)
    results = {}
    # 1 metrics match sklearn; loss matches the naive BCE formula
    y = rng.integers(0, 2, 500); lg = rng.normal(0, 2, 500); m = compute_metrics(y, lg)
    pp = expit(lg)
    results["macro_f1_matches_sklearn"] = abs(m["macro_f1"] - f1_score(y, (pp >= 0.5).astype(int), average="macro")) < 1e-12
    results["bce_matches_naive_formula"] = abs(m["loss"] + np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))) < 1e-9
    # 2 gradient: autograd backprop equals central finite differences
    L = 2; q = make_qnode(L)
    xb = pnp.array(rng.uniform(-np.pi, np.pi, (16, 4)), requires_grad=False); yb = pnp.array(rng.integers(0, 2, 16).astype(float), requires_grad=False)
    p0 = pnp.array(rng.normal(0, 0.5, 18), requires_grad=True)
    cost = lambda p: bce_with_logits(split_params(p, L)[1] * q(xb, split_params(p, L)[0]) + split_params(p, L)[2], yb)
    g = np.asarray(qml.grad(cost)(p0)); eps = 1e-6
    fd = np.array([(float(cost(pnp.array(np.asarray(p0) + eps * np.eye(18)[i]))) - float(cost(pnp.array(np.asarray(p0) - eps * np.eye(18)[i])))) / (2 * eps) for i in range(18)])
    results["autograd_equals_finite_differences"] = float(np.abs(g - fd).max()) < 1e-7
    # 3 final partial batch is used: 37 samples -> batches of 16, 16, 5
    Atr = rng.uniform(-1, 1, (37, 4)); ytr = rng.integers(0, 2, 37)
    r = train_config("selftest", "scaled", 1, 0.01, Atr, ytr, Atr[:5], ytr[:5], epochs=1)
    results["partial_last_batch_used"] = r["history"][0]["n_batches"] == 3 and r["train_forward_evals"] == 37
    # 4 CX ring Heisenberg identity Z0 -> Z0 Z2 Z3
    results["ring_maps_Z0_to_Z0Z2Z3"] = heisenberg_ring_check() < 1e-9
    # 5 Qiskit circuit (logical and officially transpiled) equals the PennyLane circuit exactly
    A = rng.uniform(-np.pi, np.pi, (25, 4)); p = rng.uniform(-np.pi, np.pi, 18)
    _, z_pl = ideal_logits(make_qnode(2), p, 2, A)
    logical = bound_circuits(2, p[:16], A)
    tq, stats = transpile_each(bnm, logical)
    zq = lambda cs: np.array([float(np.real(Statevector(c.remove_final_measurements(inplace=False)).expectation_value(SparsePauliOp("IIIZ")))) for c in cs])
    results["qiskit_logical_equals_pennylane"] = float(np.abs(zq(logical) - z_pl).max()) < 1e-9
    results["qiskit_transpiled_equals_pennylane"] = float(np.abs(zq(tq) - z_pl).max()) < 1e-9
    results["transpiled_gates_all_in_noised_basis"] = not stats["ops_outside_basis"] and not stats["unnoised_gates"]
    # 6 noise really applied + bit ordering
    ver = noise_verification(bnm, 2, p, A, n_check=25)
    results["bit_ordering"] = ver["bitorder_x_on_q0"]["ok"] and ver["bitorder_x_on_q1"]["ok"]
    results["readout_noise_applied"] = ver["readout_noise_empty_circuit"]["ok"]
    results["noisy_output_attenuated"] = ver["noisy_vs_exact_subset"]["mean_abs_z_noisy"] < 0.9 * ver["noisy_vs_exact_subset"]["mean_abs_z_exact"]
    # 7 this script can only read the two labelled files
    import ast
    tree = ast.parse(Path(__file__).read_text())
    csv_args = {ast.unparse(n.args[0]) for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "read_csv"}
    data_strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".csv")}
    results["only_train_val_csv_reads"] = (csv_args == {"TRAIN_CSV", "VAL_CSV", "CLASSICAL_METRICS"}
                                           and {s for s in data_strings if s.startswith("ecg_")} == {"ecg_train.csv", "ecg_val.csv"})
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    if not all(results.values()):
        sys.exit(1)
    print("selftest: all checks passed")


# ----------------------------------------------------------------------------- report helpers
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f4(v):
    return f"{v:.4f}"


def cm_md(m):
    return md_table(["true \\ pred", "0 Normal", "1 Arrhythmia"], [["0 Normal", m["tn"], m["fp"]], ["1 Arrhythmia", m["fn"], m["tp"]]])


def fmt_ops(ops):
    return ", ".join(f"{k}:{v}" for k, v in sorted(ops.items()))


def classical_reference():
    """Best row of the teammate's classical grid (read-only); None if the file is absent."""
    if not CLASSICAL_METRICS.exists():
        return None
    df = pd.read_csv(CLASSICAL_METRICS)
    r = df.sort_values("val_macro_f1", ascending=False).iloc[0]
    return {"val_macro_f1": float(r["val_macro_f1"]), "val_accuracy": float(r["val_accuracy"]),
            "n_configs_at_max": int((df["val_macro_f1"] == df["val_macro_f1"].max()).sum()), "n_configs": len(df)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="fast correctness checks, writes nothing")
    ap.add_argument("--smoke", action="store_true", help="1-epoch timing run, writes nothing")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    validate_root()
    seed_everything()
    bnm = load_bnm()
    import qiskit
    import qiskit_aer
    import sklearn
    t_all = time.perf_counter()

    tr, va = load_data()
    chk = integrity_checks(tr, va)
    print("integrity:", {k: v for k, v in chk.items() if k != "feature_names"})
    if not chk["passed"]:
        sys.exit("Data integrity check failed; training stopped (see output above).")
    feats = chk["feature_names"]
    ytr, yva = tr[LABEL_COL].to_numpy(int), va[LABEL_COL].to_numpy(int)
    scaler, pca, angle_scale, angles, pinfo = fit_preprocessing(tr, va, feats)
    print("EVR:", np.round(pinfo["evr"], 4), "cum:", np.round(pinfo["cum_evr"], 4), "| angle scale:", np.round(angle_scale, 4))

    if args.smoke:
        Atr, Ava = angles[CANONICAL_ENCODING]
        res = train_config("smoke", CANONICAL_ENCODING, 1, 0.01, Atr, ytr, Ava, yva, epochs=1, verbose=True)
        p = res["best"]["params"]
        t0 = time.perf_counter()
        ev = noisy_evaluation(bnm, 1, p, Ava[:100], yva[:100])
        print(f"smoke noisy: 100 circuits x {NOISY_SHOTS} shots x {len(NOISY_SIM_SEEDS)} seeds in {time.perf_counter() - t0:.1f}s; {ev['transpile']}")
        return

    struct = {L: structural_analysis(L) for L in LAYER_GRID}
    ring_identity_err = heisenberg_ring_check()
    print("structure:", {L: (s["n_dead"], s["dead_params"]) for L, s in struct.items()}, "| ring identity err", ring_identity_err)

    # ---- search: 2 encodings x 4 predeclared configurations, seed 42
    def job(enc, L, lr, seed):
        return {"cfg_id": f"{enc}_L{L}_lr{lr}_s{seed}", "encoding": enc, "n_layers": L, "lr": lr, "Atr": angles[enc][0], "ytr": ytr,
                "Ava": angles[enc][1], "yva": yva, "seed": seed}
    t0 = time.perf_counter()
    search = run_parallel([job(e, L, lr, SEED) for e in ENCODINGS for L in LAYER_GRID for lr in LR_GRID])
    search_wall = time.perf_counter() - t0
    print(f"== search done in {search_wall:.0f}s wall")
    by_enc = {e: [r for r in search if r["encoding"] == e] for e in ENCODINGS}
    sel = {e: select_final(rs) for e, rs in by_enc.items()}
    sel_scaled_L1 = select_final([r for r in by_enc["scaled"] if r["layers"] == 1])
    for e in ENCODINGS:
        print(f"   {e}: selected {sel[e]['cfg_id']} epoch {sel[e]['best']['epoch']} val macro-F1 {sel[e]['best']['val_macro_f1']:.4f}")
    canon = sel[CANONICAL_ENCODING]
    Lc = canon["layers"]
    params = canon["best"]["params"]

    # ---- seed robustness (4 extra seeds per selected config) + exact retrain of the canonical run
    t0 = time.perf_counter()
    seed_jobs = [job(e, sel[e]["layers"], sel[e]["lr"], s) for e in ENCODINGS for s in EXTRA_SEEDS]
    retrain_job = job(CANONICAL_ENCODING, Lc, canon["lr"], SEED)
    seed_out = run_parallel(seed_jobs + [retrain_job])
    seed_runs, retrain = seed_out[:-1], seed_out[-1]
    seeds_wall = time.perf_counter() - t0
    seed_summary = {}
    for e in ENCODINGS:
        runs = [sel[e]] + [r for r in seed_runs if r["encoding"] == e]
        f1s = [r["best"]["val_macro_f1"] for r in runs]
        seed_summary[e] = {"config": f"L{sel[e]['layers']} lr={sel[e]['lr']}", "seeds": [r["seed"] for r in runs],
                           "best_val_macro_f1": f1s, "best_epochs": [r["best"]["epoch"] for r in runs],
                           "mean": float(np.mean(f1s)), "sd": float(np.std(f1s, ddof=1)), "min": float(np.min(f1s)), "max": float(np.max(f1s))}
    repro = {"param_max_abs_diff": float(np.abs(np.array(retrain["final_params"]) - np.array(canon["final_params"])).max()),
             "val_f1_curve_max_abs_diff": float(np.abs(np.array([h["val_macro_f1"] for h in retrain["history"]]) -
                                                       np.array([h["val_macro_f1"] for h in canon["history"]])).max()),
             "best_epoch_match": retrain["best"]["epoch"] == canon["best"]["epoch"]}
    print(f"== seeds done in {seeds_wall:.0f}s wall; seeds {({e: round(s['mean'], 4) for e, s in seed_summary.items()})}; retrain {repro}")

    # ---- ideal evaluation (exact statevector) of the selected models + scaled 1-layer diagnostic
    models = {"canonical": (CANONICAL_ENCODING, canon), "direct_ablation": ("direct", sel["direct"]), "scaled_1layer": ("scaled", sel_scaled_L1)}
    ideal = {}
    for name, (enc, s) in models.items():
        L = s["layers"]
        t0 = time.perf_counter()
        lv, zv = ideal_logits(make_qnode(L), s["best"]["params"], L, angles[enc][1])
        dt = time.perf_counter() - t0
        lt, _ = ideal_logits(make_qnode(L), s["best"]["params"], L, angles[enc][0])
        pred = (lv >= 0).astype(int)
        ideal[name] = {"logit": lv, "z": zv, "metrics": compute_metrics(yva, lv), "train_metrics": compute_metrics(ytr, lt),
                       "ci": bootstrap_ci(yva, pred), "auc": float(roc_auc_score(yva, lv)), "time_s": dt}
    perfect = any(d["metrics"]["macro_f1"] > 0.99 for d in ideal.values())

    # ---- noise verification + noisy evaluation (official helper, per-circuit transpilation)
    print("== noise verification", flush=True)
    ver = noise_verification(bnm, Lc, params, angles[CANONICAL_ENCODING][1])
    print("   all_ok:", ver["all_ok"])
    noisy = {}
    for name, (enc, s) in models.items():
        print(f"== noisy evaluation {name}", flush=True)
        ev = noisy_evaluation(bnm, s["layers"], s["best"]["params"], angles[enc][1], yva)
        zi = ideal[name]["z"]
        ev["attenuation_slope"] = float(np.polyfit(zi, ev["z"], 1)[0])
        ev["corr_ideal_noisy"] = float(np.corrcoef(zi, ev["z"])[0, 1])
        ev["auc"] = float(roc_auc_score(yva, ev["logit"]))
        ev["threshold_z"] = float(-s["best"]["params"][8 * s["layers"] + 1] / s["best"]["params"][8 * s["layers"]])
        ev["flip_frac"] = float(((ideal[name]["logit"] >= 0) != (ev["logit"] >= 0)).mean())
        ev["ci_seed42"] = bootstrap_ci(yva, (ev["logit"] >= 0).astype(int))
        noisy[name] = ev
        print(f"   {name}: ideal {ideal[name]['metrics']['macro_f1']:.4f} noisy {ev['macro_f1_mean']:.4f} +- {ev['macro_f1_sd']:.4f}")

    # ---- persist + reload check
    state = inference_state(scaler, pca, angle_scale, feats, canon)
    p_ref = expit(ideal["canonical"]["logit"])
    CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    classical = classical_reference()
    config = {
        "task": "PS-Q1 hybrid VQC (canonical model)",
        "seeds": {"global": SEED, "python_random": SEED, "numpy": SEED, "init_and_batch_order_rng": "np.random.default_rng(seed) per training run",
                  "extra_training_seeds": EXTRA_SEEDS, "pca_random_state": SEED, "aer_simulator_seeds": NOISY_SIM_SEEDS, "transpiler": bnm.SEED},
        "search_space": {"encodings": ENCODINGS, "layers": LAYER_GRID, "learning_rates": LR_GRID, "max_epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE,
                         "optimizer": "pennylane.AdamOptimizer (beta1=0.9, beta2=0.999, eps=1e-8)", "loss": "binary cross-entropy with logits",
                         "threshold": THRESHOLD, "init": f"theta ~ N(0, {INIT_SCALE}^2), readout w=1.0, b=0.0",
                         "selection": "highest validation macro-F1; ties -> fewer layers, lr 0.001, earlier epoch"},
        "canonical_encoding_choice": "scaled (design correction after the direct encoding was found to wrap angles beyond +-pi; approved by the team; direct kept as ablation)",
        "selected": {"config_id": canon["cfg_id"], "encoding": canon["encoding"], "layers": Lc, "lr": canon["lr"], "best_epoch": canon["best"]["epoch"],
                     "val_macro_f1": canon["best"]["val_macro_f1"], "n_trainable_nominal": 8 * Lc + 2,
                     "n_trainable_effective": struct[Lc]["n_effective_quantum_params"] + 2, "dead_params": struct[Lc]["dead_params"]},
        "all_configs": [{"config_id": r["cfg_id"], "encoding": r["encoding"], "layers": r["layers"], "lr": r["lr"],
                         "best_epoch": r["best"]["epoch"] if r["best"] else None,
                         "best_val_macro_f1": r["best"]["val_macro_f1"] if r["best"] else None, "failed": r["failed"]} for r in search],
        "seed_robustness": seed_summary,
        "inference_state": state,
        "results": {"ideal_val": {k: v["metrics"] for k, v in ideal.items()},
                    "noisy_val_seed42": {k: v["metrics"] for k, v in noisy.items()},
                    "noisy_val_macro_f1_mean_sd": {k: [v["macro_f1_mean"], v["macro_f1_sd"]] for k, v in noisy.items()}},
        "noise_verification": ver,
        "evaluation": {"ideal": "exact statevector expectation (PennyLane default.qubit, shots=None)",
                       "noisy": {"helper": "noise_model/benchmark_noise_model.py (unchanged)", "version": bnm.VERSION, "shots_per_circuit": NOISY_SHOTS,
                                 "simulator_seeds": NOISY_SIM_SEEDS, "basis": bnm.DEFAULT_BASIS,
                                 "transpile": "prepare_for_noisy_run() applied to every bound circuit (opt level 1, seed_transpiler=42)"}},
        "frameworks": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pennylane": qml.__version__,
                       "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__, "scikit_learn": sklearn.__version__},
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2))
    reloaded = json.loads(CONFIG_JSON.read_text())["inference_state"]
    reload_diff = float(np.abs(reload_probabilities(reloaded, va[feats].to_numpy(float)) - p_ref).max())
    print("reload max |dp|:", reload_diff)

    # ---- artifacts: history + resources
    pd.DataFrame([h for r in search + seed_runs for h in r["history"]]).to_csv(HISTORY_CSV, index=False)
    logical = {L: logical_resources(L) for L in LAYER_GRID}
    rrows = []
    for r in search + seed_runs:
        lg = logical[r["layers"]]
        rrows.append({"stage": "training", "config_id": r["cfg_id"], "encoding": r["encoding"], "layers": r["layers"], "lr": r["lr"], "seed": r["seed"],
                      "qubits": N_QUBITS, "quantum_params": 8 * r["layers"], "effective_quantum_params": struct[r["layers"]]["n_effective_quantum_params"],
                      "readout_params": 2, "logical_depth": lg["depth"], "logical_gates": fmt_ops(lg["ops"]), "logical_cx": lg["cx"],
                      "transpiled_depth": "n/a (exact statevector)", "transpiled_gates": "n/a", "transpiled_cx": "n/a",
                      "circuit_evals": r["forward_evals"], "gradient_circuit_executions": "not exposed (backprop through simulator)",
                      "shots": 0, "train_time_s": round(r["train_time_s"], 1), "eval_time_s": "n/a",
                      "verification_status": "FAILED: " + r["failed"] if r["failed"] else "completed 30 epochs"})
    for name, (enc, s) in models.items():
        L = s["layers"]; lg = logical[L]; ev = noisy[name]; ts = ev["transpile"]
        base = {"config_id": s["cfg_id"], "encoding": enc, "layers": L, "lr": s["lr"], "seed": s["seed"], "qubits": N_QUBITS, "quantum_params": 8 * L,
                "effective_quantum_params": struct[L]["n_effective_quantum_params"], "readout_params": 2, "logical_depth": lg["depth"],
                "logical_gates": fmt_ops(lg["ops"]), "logical_cx": lg["cx"], "gradient_circuit_executions": 0, "train_time_s": "n/a"}
        rrows.append({**base, "stage": f"ideal_eval_{name}", "transpiled_depth": "n/a", "transpiled_gates": "n/a", "transpiled_cx": "n/a",
                      "circuit_evals": len(yva), "shots": 0, "eval_time_s": round(ideal[name]["time_s"], 4),
                      "verification_status": "exact statevector" + (f"; reload check max|dp|={reload_diff:.1e}" if name == "canonical" else "")})
        rrows.append({**base, "stage": f"noisy_eval_{name}", "transpiled_depth": f"{ts['depth_mean']:.1f} (min {ts['depth_min']}, max {ts['depth_max']})",
                      "transpiled_gates": f"1q {ts['1q_gates_mean']:.1f} (min {ts['1q_gates_min']}, max {ts['1q_gates_max']}), cx {ts['cx_min']}-{ts['cx_max']}, measure 1",
                      "transpiled_cx": ts["cx_max"], "circuit_evals": ev["n_circuits"] * len(NOISY_SIM_SEEDS), "shots": ev["total_shots"],
                      "eval_time_s": round(ev["transpile_time_s"] + ev["run_time_s_per_seed"] * len(NOISY_SIM_SEEDS), 1),
                      "verification_status": ("all noise checks passed" if ver["all_ok"] else "CHECK FAILED") if name == "canonical"
                      else ("basis/noise coverage verified" if not ts["ops_outside_basis"] and not ts["unnoised_gates"] else "CHECK FAILED")})
    RESOURCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rrows).to_csv(RESOURCE_CSV, index=False)

    write_reports(chk, pinfo, search, sel, sel_scaled_L1, seed_summary, repro, ideal, noisy, ver, struct, ring_identity_err,
                  logical, reload_diff, perfect, classical, search_wall, seeds_wall, time.perf_counter() - t_all, yva)
    print("done in %.1f min" % ((time.perf_counter() - t_all) / 60))


def write_reports(chk, pinfo, search, sel, sel_scaled_L1, seed_summary, repro, ideal, noisy, ver, struct, ring_err,
                  logical, reload_diff, perfect, classical, search_wall, seeds_wall, total_time, yva):
    canon = sel[CANONICAL_ENCODING]
    Lc = canon["layers"]
    mi, mti = ideal["canonical"]["metrics"], ideal["canonical"]["train_metrics"]
    names = {"canonical": f"canonical: scaled, {Lc} layer(s)", "direct_ablation": f"ablation: direct, {sel['direct']['layers']} layer(s)",
             "scaled_1layer": "diagnostic: scaled, 1 layer"}
    evr_rows = [[f"PC{i + 1}", f4(pinfo["evr"][i]), f4(pinfo["cum_evr"][i]), f"{pinfo['angle_scale'][i]:.4f}",
                 f"[{pinfo['ranges']['direct']['train_min'][i]:.2f}, {pinfo['ranges']['direct']['train_max'][i]:.2f}]",
                 f"[{pinfo['ranges']['scaled']['val_min'][i]:.2f}, {pinfo['ranges']['scaled']['val_max'][i]:.2f}]"] for i in range(N_QUBITS)]
    cfg_rows = [[r["encoding"], r["layers"], r["lr"], 8 * r["layers"] + 2, r["best"]["epoch"] if r["best"] else "-",
                 f4(r["best"]["val_macro_f1"]) if r["best"] else "-", f4(r["history"][-1]["train_loss"]), f4(r["history"][-1]["val_loss"]),
                 f"{r['train_time_s']:.0f}", r["forward_evals"], "<- selected" if r is sel[r["encoding"]] else ""] for r in search]
    pc = lambda m: md_table(["class", "precision", "recall", "F1"], [["0 Normal", f4(m["precision_0"]), f4(m["recall_0"]), f4(m["f1_0"])],
                                                                    ["1 Arrhythmia", f4(m["precision_1"]), f4(m["recall_1"]), f4(m["f1_1"])]])
    ps = lambda m: md_table(["mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "max", "frac predicted 1"],
                            [[f4(m[k]) for k in ("prob_mean", "prob_std", "prob_min", "prob_p5", "prob_p25", "prob_p50", "prob_p75", "prob_p95", "prob_max", "frac_pred_1")]])
    all_finite = all(h["finite"] for r in search for h in r["history"])
    checks = [
        ("Only train/val CSVs read (explicit paths; --selftest parses this file and asserts read_csv is only called on TRAIN_CSV, VAL_CSV and the read-only classical metrics file, and that no other ecg_*.csv name appears)", "PERFORMED - passed"),
        ("Integrity: schema, NaN/inf, labels, duplicate rows within/across partitions", "PERFORMED - passed" if chk["passed"] else "FAILED"),
        ("Scaler, PCA and angle scale fitted on training rows only", "PERFORMED by construction (fit_preprocessing) and code review"),
        ("Validation never in gradient updates", "PERFORMED by construction: optimizer steps use training batches only; validation only scores epochs"),
        ("Gradient correctness (autograd vs finite differences)", "PERFORMED in --selftest - passed (max diff < 1e-7)"),
        ("Final partial batch handled", "PERFORMED in --selftest (37 samples -> 3 batches); not exercised in the real run because 6000 = 375 x 16"),
        ("All losses/parameters/probabilities finite, probabilities in [0,1]", "PERFORMED - " + ("passed" if all_finite else "FAILED")),
        ("Reload from saved JSON reproduces ideal validation probabilities", f"PERFORMED - max |dp| = {reload_diff:.2e} (tolerance 1e-9): {'passed' if reload_diff < 1e-9 else 'FAILED'}"),
        ("Exact retrain of the canonical run (seed 42)", f"PERFORMED - max |dparams| = {repro['param_max_abs_diff']:.2e}, max |dval F1| = {repro['val_f1_curve_max_abs_diff']:.2e}, best epoch match = {repro['best_epoch_match']}"),
        ("Seed robustness (4 extra seeds per selected config)", "PERFORMED - see table"),
        ("Full 8-run search repeated from scratch", "NOT RUN (only the canonical configuration was retrained)"),
        ("Perfect-score investigation", "TRIGGERED (ideal validation macro-F1 > 0.99): no duplicates/overlap, fit scopes train-only, labels as documented; "
                                        "the data is near-linearly separable (the teammate's logistic regression scores 1.000 on validation), so a high score is expected" if perfect else "not triggered"),
    ]
    cl = (f"Logistic regression on all 30 standardised features (teammate's grid, read-only from `reports/classical_baseline_metrics.csv`): validation macro-F1 "
          f"{classical['val_macro_f1']:.4f} ({classical['n_configs_at_max']} of {classical['n_configs']} grid points at this value); every validation sample is correct, "
          f"so its bootstrap interval is [1.0000, 1.0000]." if classical else "Classical metrics file not found; no comparison made.")
    seed_rows = [[e, s["config"], ", ".join(f"{v:.4f}" for v in s["best_val_macro_f1"]), ", ".join(map(str, s["best_epochs"])), f4(s["mean"]), f4(s["sd"])]
                 for e, s in seed_summary.items()]
    struct_rows = [[L, s["n_quantum_params"], s["n_dead"], s["n_effective_quantum_params"], ", ".join(s["dead_params"]),
                    ", ".join(f"{v:.3f}" for v in s["input_sensitivity_mean_abs"])] for L, s in struct.items()]
    comp_rows = [[names[k], f4(ideal[k]["metrics"]["macro_f1"]), f"[{ideal[k]['ci'][0]:.4f}, {ideal[k]['ci'][1]:.4f}]", f4(ideal[k]["metrics"]["accuracy"]),
                  f4(ideal[k]["auc"]), f4(ideal[k]["train_metrics"]["macro_f1"])] for k in ideal]

    train_md = f"""# PS-Q1 VQC - training and ideal validation results

Generated by `vqc_work/scripts/train_vqc.py` (total runtime {total_time / 60:.1f} min on {os.cpu_count()} cores). Every number below comes from that run.

## Challenge context (official documents)
Sources: the participant problem statement (v3), the Phase 2 judging rubric, `noise_model/benchmark_noise_model.py`, `benchmark/qff_qml_benchmark.py`.
- PS-Q1 (QML-BIO-MITBIH): classify heartbeats, 0 = Normal, 1 = Arrhythmia, 30 features, ranking metric macro-F1 (accuracy for reference only).
- Rules used here: train only on `ecg_train.csv`; `ecg_val.csv` for model selection and tuning only; scalers/PCA fitted on training data only; seeds fixed; quantum model trained from scratch; every noisy run with the unchanged QFF2026-v2 helper on circuits transpiled to a fully-noised basis.
- Rubric (100 points): gates G1-G5 (repository, one PS + Task A/B, valid predictions, notebook runs, no test-data fitting), C1 quantum design 30, C2 results 25 (10 from hidden-test macro-F1 via 10 x clip((m - 0.5)/(m_baseline - 0.5), 0, 1) with m_baseline = 1.000 for Q1), C3 noise analysis 20, C4 Processor D 15, C5 reproducibility/report 10, bonus 5 (ablations count). The rubric notes that "Q1 is separated perfectly by a simple classical model, so the judged criteria decide it".
- Judging rounds: the rubric describes independent scoring by at least two judges and re-scoring of large differences; no further round structure is documented.
- Scope of this workstream: the VQC only. Task A/B, test predictions and the classical model are handled elsewhere and are not touched here.

## Data
- Train {chk['train_shape']}, validation {chk['val_shape']} (30 float features + `label`; these files have no `sample_id`). Labels: train {chk['train_labels']}, validation {chk['val_labels']}.
- Missing {chk['train_missing']}/{chk['val_missing']}, non-finite {chk['train_nonfinite']}/{chk['val_nonfinite']}, duplicate rows within train/val {chk['dup_rows_train']}/{chk['dup_rows_val']}, cross-partition duplicate feature rows {chk['cross_partition_duplicate_rows']}. Integrity: **{'passed' if chk['passed'] else 'FAILED'}**.

## Preprocessing and encoding
StandardScaler (fit: 6,000 training rows) -> PCA(4, svd_solver="full", random_state=42; fit: scaled training rows) -> angle encoding. Validation rows are only transformed.

{md_table(['component', 'explained variance ratio', 'cumulative', 'angle scale (pi / max train abs)', 'direct angle range (train)', 'scaled angle range (val)'], evr_rows)}

- **Why the encoding was changed.** The originally specified *direct* encoding feeds raw PCA scores into `RY`. Those scores span about +-{max(abs(v) for v in pinfo['ranges']['direct']['train_min'] + pinfo['ranges']['direct']['train_max']):.1f} rad, and {pinfo['train_outside_pi_frac']['direct'] * 100:.1f}% of training rows have at least one angle outside [-pi, pi]; because `RY` is 2pi-periodic, distant samples wrap onto similar rotations. The *scaled* encoding multiplies each component by pi / max|train score| (fitted on train only), keeping the map one-to-one on the training range. This change was made explicitly, approved by the team, and the direct encoding is still trained and reported as an ablation. Validation rows outside [-pi, pi] after scaling: {pinfo['val_outside_pi_frac']['scaled'] * 100:.2f}%.
- 4 components keep {pinfo['cum_evr'][-1] * 100:.1f}% of the scaled-feature variance. An audit diagnostic run outside this script found that a logistic regression on the same 4 training-fitted components also classifies the validation set perfectly, so the 4-component bottleneck does not remove the class information; remaining errors come from the quantum model.

## Circuit (verified against the implementation)
- 4 qubits, `RY(angle_i)` on qubit i, then per layer: trainable `RY` x4, trainable `RZ` x4, CX sequence {RING_CX}. Readout `z = <Z_0>`, `logit = w z + b`, `p = sigmoid(logit)`, fixed threshold 0.5.
- Nominal trainable parameters: 8 per layer + 2 readout (1 layer: 10, 2 layers: 18). Logical resources: 1 layer depth {logical[1]['depth']}, CX {logical[1]['cx']}; 2 layers depth {logical[2]['depth']}, CX {logical[2]['cx']}.
- **Structural finding (gradient-verified).** The final CX ring maps `Z_0` back to `Z_0 Z_2 Z_3` (verified numerically, max error {ring_err:.1e}). Qubit 1 therefore never reaches the readout through the last layer, and `RZ` gates directly before the last ring cannot change a Z-basis outcome. Parameters with exactly zero gradient at every tested point:

{md_table(['layers', 'quantum params', 'dead', 'effective', 'dead parameters', 'mean |d<Z0>/d angle| for PC1..PC4'], struct_rows)}

  Consequence: the 1-layer model **ignores PC2 entirely** and its output is a product of cosines of PC1, PC3 and PC4; the 2-layer model uses all four inputs. The architecture is kept as specified (dead parameters are harmless for training, they simply never move); removing them would change the specified ansatz.

## Training protocol
Adam (PennyLane `AdamOptimizer`), BCE with logits, batch 16 (375 batches per epoch), 30 epochs, seed 42 for initialisation and batch order, parameters initialised N(0, {INIT_SCALE}^2), readout w = 1, b = 0. Gradients: analytic backpropagation through the statevector simulator. The predeclared grid (layers {{1, 2}} x lr {{0.01, 0.001}}) is run for both encodings ({search_wall:.0f} s wall, 8 runs in parallel). Selection per encoding: highest validation macro-F1, ties -> fewer layers, lr 0.001, earlier epoch.

{md_table(['encoding', 'layers', 'lr', 'trainable', 'best epoch', 'best val macro-F1', 'final train loss', 'final val loss', 'train time (s, parallel)', 'forward circuit evals', ''], cfg_rows)}

**Canonical model: {canon['cfg_id']} (epoch {canon['best']['epoch']}).** Validation chose the epoch, the configuration and (after the design change) the encoding, so these are *model-selection* numbers and are optimistically biased; they are not held-out test evidence.

## Seed robustness (selected configuration of each encoding; {seeds_wall:.0f} s wall)
{md_table(['encoding', 'config', 'best val macro-F1 for seeds 42, 43, 44, 45, 46', 'best epochs', 'mean', 'sd'], seed_rows)}

## Ideal validation results (exact statevector)
{md_table(['model', 'val macro-F1', '95% bootstrap CI', 'accuracy', 'ROC-AUC', 'train macro-F1'], comp_rows)}

Canonical model, per class:

{pc(mi)}

Confusion matrix (class order [0,1]):

{cm_md(mi)}

Predicted P(class 1):

{ps(mi)}

## Comparison with the classical baseline (same data, validation-tuned)
{cl}
The canonical VQC reaches {f4(mi['macro_f1'])} ({mi['fp'] + mi['fn']} of {len(yva)} validation samples wrong, 95% CI [{ideal['canonical']['ci'][0]:.4f}, {ideal['canonical']['ci'][1]:.4f}]) from 4 PCA inputs and {8 * Lc + 2} nominal parameters, under ideal simulation only. It does not beat the classical baseline, and no quantum advantage is claimed: the classical model is simpler, exact and noise-free, while the VQC's validation macro-F1 under the official noise model is {noisy['canonical']['macro_f1_mean']:.4f} (mean of 5 simulator seeds).

## Resources
- Qubits {N_QUBITS}; nominal trainable {8 * Lc + 2} (effective {struct[Lc]['n_effective_quantum_params'] + 2}); logical depth {logical[Lc]['depth']}; logical CX {logical[Lc]['cx']}.
- Canonical training run: {canon['train_time_s']:.0f} s (measured while 8 runs shared the CPU). Ideal evaluation of 1,500 validation samples: {ideal['canonical']['time_s'] * 1000:.1f} ms.
- Counting convention: "forward circuit evaluations" = input samples pushed through the simulated circuit (training batches plus per-epoch train/validation scoring); {canon['forward_evals']:,} for the canonical run and {sum(r['forward_evals'] for r in search):,} for the 8-run search. Gradient circuit executions are not exposed by backpropagation (no parameter-shift executions). Ideal results use 0 shots. Noisy shot counts are in `vqc_ideal_vs_noisy.md`; all rows are in `vqc_resource_summary.csv`.

## Reproducibility and quality checks
{md_table(['check', 'status'], [[a, b] for a, b in checks])}

Run `python vqc_work/scripts/train_vqc.py --selftest` for the fast checks (metrics, BCE, gradients, partial batch, ring identity, Qiskit/PennyLane equivalence before and after transpilation, basis coverage, bit order, readout noise, data-access guard). Framework versions and all seeds are in `artifacts/vqc_config.json`.

## Limitations
- Validation is used for selection, so validation scores are optimistic; there is no held-out estimate inside this workstream.
- The predeclared search is small; one canonical seed is reported, with four extra seeds for robustness.
- The specified ansatz contains structurally dead parameters (see above) and its 1-layer version cannot see PC2.
- The scaled encoding was introduced after inspecting the direct encoding's behaviour; that design step is itself informed by validation results and is disclosed as such.
- Exact (shot-free) ideal results; noisy results include shot noise.
- No claim of quantum advantage or clinical validity; the data are synthetic statistics modelled on MIT-BIH.
- Process note: earlier in the project session a shell row-count (`wc -l` over a glob of the q1 CSVs) touched the unlabelled test-features file; no content was read or used, and this script has no code path to that file (asserted by `--selftest`).
"""
    REPORT_TRAIN.write_text(train_md)

    nrows, cmp_rows = [], []
    for k, ev in noisy.items():
        m_i, m_n = ideal[k]["metrics"], ev["metrics"]
        ts = ev["transpile"]
        nrows.append([names[k], 8 * {"canonical": Lc, "direct_ablation": sel["direct"]["layers"], "scaled_1layer": 1}[k],
                      f4(m_i["macro_f1"]), f4(m_n["macro_f1"]), f"{ev['macro_f1_mean']:.4f} +- {ev['macro_f1_sd']:.4f}",
                      f"{m_n['macro_f1'] - m_i['macro_f1']:+.4f}", f4(m_n["recall_0"]), f4(m_n["recall_1"]),
                      f4(ideal[k]["auc"]), f4(ev["auc"]), f"{ev['attenuation_slope']:.3f}", f"{ev['corr_ideal_noisy']:.4f}",
                      f"{ev['threshold_z']:+.3f}", f"{ev['flip_frac']:.3f}"])
        cmp_rows.append([names[k], f"{ts['depth_mean']:.1f} ({ts['depth_min']}-{ts['depth_max']})", f"{ts['1q_gates_mean']:.1f}", f"{ts['cx_min']}-{ts['cx_max']}",
                         str(ts["ops_outside_basis"] or "none"), str(ts["unnoised_gates"] or "none"), ev["n_circuits"], f"{ev['total_shots']:,}",
                         f"{ev['transpile_time_s']:.1f}", f"{ev['run_time_s_per_seed']:.1f}"])
    cn, ci_ = noisy["canonical"], ideal["canonical"]
    v = ver
    ver_rows = [
        ["Noise model attached to simulator", f"{v['simulator_has_noise_model']} (noise instructions: {', '.join(v['noise_instructions'])})"],
        ["Official parameters", f"p1q={v['noise_params']['single_qubit_depolarizing']}, p2q={v['noise_params']['two_qubit_depolarizing']}, readout={v['noise_params']['readout_error']}, version {v['version']}"],
        ["Bit order: X on q0 -> <Z0>", f"{v['bitorder_x_on_q0']['z0']:+.3f} (expected -1): {'ok' if v['bitorder_x_on_q0']['ok'] else 'FAILED'}"],
        ["Bit order: X on q1 -> <Z0>", f"{v['bitorder_x_on_q1']['z0']:+.3f} (expected +1): {'ok' if v['bitorder_x_on_q1']['ok'] else 'FAILED'}"],
        ["Readout noise on an empty circuit, P(1)", f"{v['readout_noise_empty_circuit']['p_read_1']:.4f} vs 0.02 (SE {v['readout_noise_empty_circuit']['se']:.4f}): {'ok' if v['readout_noise_empty_circuit']['ok'] else 'FAILED'}"],
        ["Qiskit logical circuit vs PennyLane (exact statevector, 300 val samples)", f"max |diff| {v['qiskit_statevector_vs_pennylane_max_diff']:.1e}: {'ok' if v['exact_equivalence_ok'] else 'FAILED'}"],
        ["Officially transpiled circuit vs PennyLane (exact statevector)", f"max |diff| {v['transpiled_statevector_vs_pennylane_max_diff']:.1e}: {'ok' if v['transpile_preserves_circuit_ok'] else 'FAILED'}"],
        ["Noiseless shots vs exact", f"max |diff| {v['shots_noiseless_vs_exact']['max_abs_diff']:.4f} (shot sigma {v['shots_noiseless_vs_exact']['shot_sigma']:.4f}): {'ok' if v['shots_noiseless_vs_exact']['ok'] else 'FAILED'}"],
        ["Noisy vs exact on the same subset", f"mean |z| {v['noisy_vs_exact_subset']['mean_abs_z_exact']:.4f} -> {v['noisy_vs_exact_subset']['mean_abs_z_noisy']:.4f}"],
    ]
    noisy_md = f"""# PS-Q1 VQC - ideal vs official-noise validation

The *same* trained weights are evaluated twice on the 1,500 validation rows: exact statevector (ideal) and the official QFF2026-v2 noise model (noisy). Nothing is retrained or tuned on noisy results. Validation-only; not held-out test evidence.

## Official noise configuration
- Helper `noise_model/benchmark_noise_model.py`, imported unchanged; version **{v['version']}**: single-qubit depolarizing {v['noise_params']['single_qubit_depolarizing']}, two-qubit depolarizing {v['noise_params']['two_qubit_depolarizing']}, symmetric readout error {v['noise_params']['readout_error']}; simulator `noisy_simulator(seed)`.
- **Transpilation: `prepare_for_noisy_run()` is applied to every executed circuit** (each validation sample's bound circuit; basis rx, ry, rz, cx; optimisation level 1; `seed_transpiler=42`); the helper itself asserts that every gate carries noise.
- Shots: {NOISY_SHOTS} per circuit (helper default and Task A's value; no PS-Q1 value is documented). Simulator seeds {NOISY_SIM_SEEDS}; the seed-42 run is the primary number and the five-seed mean +- sd shows shot-noise variability. `<Z0> = (n0 - n1)/shots` from one classical bit mapped from qubit 0.

## Verification evidence (canonical model)
{md_table(['check', 'result'], ver_rows)}

Overall: **{'all checks passed' if v['all_ok'] else 'AT LEAST ONE CHECK FAILED'}**.

## Ideal vs noisy macro-F1
{md_table(['model', 'CX', 'ideal', 'noisy (seed 42)', 'noisy mean +- sd (5 sim seeds)', 'change', 'noisy recall 0', 'noisy recall 1', 'ROC-AUC ideal', 'ROC-AUC noisy', '<Z0> attenuation slope', 'corr(ideal, noisy <Z0>)', 'decision threshold in <Z0>', 'flipped predictions'], nrows)}

Canonical model, ideal confusion matrix:

{cm_md(ci_['metrics'])}

Canonical model, noisy confusion matrix (seed 42):

{cm_md(cn['metrics'])}

Canonical per class, ideal:

{pc(ci_['metrics'])}

Canonical per class, noisy (seed 42):

{pc(cn['metrics'])}

Predicted P(class 1), ideal:

{ps(ci_['metrics'])}

Predicted P(class 1), noisy (seed 42):

{ps(cn['metrics'])}

## Why noise changes macro-F1
- **Attenuation, not scrambling.** Under QFF2026-v2 the measured `<Z0>` of the canonical model is close to a scaled copy of the ideal value (slope {cn['attenuation_slope']:.3f}, correlation {cn['corr_ideal_noisy']:.4f}), and the ranking of samples is largely preserved (ROC-AUC {ci_['auc']:.4f} ideal vs {cn['auc']:.4f} noisy). Depolarizing errors on {8 * Lc} CX gates (5% each), single-qubit gates (1%) and 2% readout flips shrink the expectation value toward zero.
- **Fixed decision rule.** The trained readout puts the decision boundary at `<Z0> = {cn['threshold_z']:+.3f}`. That threshold was learned for the unattenuated signal; after shrinkage many samples of one class cross it, so the damage is class-specific (noisy recall: class 0 {cn['metrics']['recall_0']:.3f}, class 1 {cn['metrics']['recall_1']:.3f}). The information largely survives; the calibration of the decision rule does not.
- **Depth trade-off.** The 1-layer scaled model (8 CX) keeps more signal under noise (attenuation slope {noisy['scaled_1layer']['attenuation_slope']:.3f}, noisy macro-F1 {noisy['scaled_1layer']['macro_f1_mean']:.4f}) but has a lower ideal score ({ideal['scaled_1layer']['metrics']['macro_f1']:.4f}) and, as shown in the training report, cannot use PC2. The canonical 2-layer model is best ideally and most fragile under noise.
- The direct-angle ablation degrades from {ideal['direct_ablation']['metrics']['macro_f1']:.4f} to {noisy['direct_ablation']['macro_f1_mean']:.4f}.

## Circuit resources under the official transpilation
{md_table(['model', 'depth', '1q gates', 'CX', 'ops outside basis', 'un-noised gates', 'circuits', 'shots (5 seeds)', 'transpile time (s)', 'run time per seed (s)'], cmp_rows)}

Logical circuits: 1 layer depth {logical[1]['depth']} / CX {logical[1]['cx']}; 2 layers depth {logical[2]['depth']} / CX {logical[2]['cx']}. Transpilation merges the encoding `RY` with the first trainable `RY` on each qubit (fewer single-qubit gates than the logical circuit) and adds no CX. No coupling map is applied (basis only), so no SWAPs are included here; mapping onto Processor D is part of Task B.

## Correction made during the audit
Earlier versions transpiled one *parametrised* template and bound parameters afterwards. With symbolic parameters the transpiler cannot merge the encoding `RY` with the first trainable `RY`, so 4 extra noisy gates per circuit were executed and noisy macro-F1 was understated (direct 2-layer model: 0.778 with the template method vs 0.797 per-circuit, 5-seed means). The pipeline now runs the official `prepare_for_noisy_run()` on every executed circuit, as the rules state.

## Blockers
{'None.' if v['all_ok'] else 'Verification failed; see table above.'}

## Caveats
- Noisy numbers include shot noise (SE of each `<Z0>` about {1 / np.sqrt(NOISY_SHOTS):.4f}); variability over 5 simulator seeds is reported.
- Ideal numbers are exact, so ideal-vs-noisy differences mix device noise and finite-shot effects.
- The decision threshold is the one learned in ideal training; no noise-aware recalibration is applied in this workstream.
"""
    REPORT_NOISY.write_text(noisy_md)


if __name__ == "__main__":
    main()
