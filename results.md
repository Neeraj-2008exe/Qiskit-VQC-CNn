# QFF2026 QML Track — Results Report

**Team name:** _TEAM NAME_
**Selected PS:** [x] PS-Q1  [ ] PS-Q2  [ ] PS-Q3
**Date:** 2026-10-30
**Dependency versions:** pennylane==0.44.1, PennyLane-qiskit==0.44.1, qiskit==2.3.0, qiskit-aer==0.17.2, scikit-learn==1.9.1, numpy==2.4.6, pandas==3.1.0 (Python 3.13.13; full list in README.md)
**Noise model version:** QFF2026-v2

All numbers below are produced by `solution.ipynb` (and, identically, by `vqc_work/scripts/train_vqc.py`, which was run twice from scratch with identical results). All PS-Q1 metrics are on the validation set, which was also used for model selection, so they are optimistic; the hidden test set was used only for the final predictions.

---

## 1. Approach Summary

We classify heartbeats with a 4-qubit hybrid classifier: the 30 window statistics are standardised and reduced to 4 principal components (fitted on training rows only), each component is rescaled to [-π, π] with a training-fitted scale and loaded as an `RY` angle, two trainable layers of `RY`+`RZ` rotations with a bidirectional CX ring (CX(0,1) CX(1,0) CX(1,2) CX(2,1) CX(2,3) CX(3,2) CX(3,0) CX(0,3)) process it, and ⟨Z₀⟩ goes through a trainable affine map and a sigmoid. The angle rescaling is the key design decision: raw PCA scores reach ±8.6 rad and 52.8% of training rows have an angle outside [-π, π], so the 2π-periodic `RY` wraps distant samples together; rescaling lifts validation macro-F1 from 0.9633 to 0.9993 for the same circuit. The bidirectional 4-qubit ring is matched by our Processor D, so the circuit runs there without routing. We also analysed what the ansatz can learn: the final ring maps Z₀ to Z₀Z₂Z₃, so 6 of the 16 rotation angles never receive a gradient and a 1-layer version cannot see PC2 at all, which is why two layers are needed.

---

## 2. Results

### Classical baseline
| Model | Official metric | Notes |
|-------|----------------|-------|
| Logistic regression, 30 standardised features | macro-F1 **1.0000** (validation, 0 errors) | Grid C ∈ {0.01, 0.1, 1, 10, 100} × class_weight ∈ {None, balanced}, all 10 points give 1.0000; selected C=1, no weighting; scaler fit on train only, selection on validation only |

### Quantum / hybrid model
Official metric: Q1 macro-F1. Ideal = exact statevector; noisy = official QFF2026-v2 model, `prepare_for_noisy_run()` on every executed circuit, 8192 shots, simulator seeds 42–46 (mean ± sd).

| Configuration | Ideal | Noisy (benchmark) | Notes |
|---------------|-------|--------------------|-------|
| **Submitted: scaled angles, 2 layers, lr 0.001, epoch 19** | **0.9993** (95% bootstrap CI 0.9979–1.0000; 1 error of 1500) | 0.4663 ± 0.0048 | Same 0.9993 for all 5 training seeds. Noise shrinks ⟨Z₀⟩ to 0.40× but keeps the ranking (ROC-AUC 1.0000 ideal and noisy); the fixed threshold then calls most Normal beats Arrhythmia (noisy recall: Normal 0.192, Arrhythmia 1.000) |
| Ablation: direct (unscaled) angles, 2 layers, lr 0.01 | 0.9633 (CI 0.9534–0.9727) | 0.7971 ± 0.0037 | 5 training seeds: 0.9636 ± 0.0008 |
| Diagnostic: scaled angles, 1 layer, lr 0.01 | 0.9667 (CI 0.9570–0.9751) | 0.9496 ± 0.0013 | 8 CX instead of 16: shrinkage 0.60× instead of 0.40×, far more robust; cannot use PC2 |
| Predeclared grid, all 8 runs | scaled: 0.9667 / 0.9660 / 0.9993 / 0.9993; direct: 0.8855 / 0.8806 / 0.9633 / 0.9511 | — | (1 layer lr 0.01 / 1 layer lr 0.001 / 2 layers lr 0.01 / 2 layers lr 0.001) |

Test predictions (`predictions_q1.csv`) come from the submitted model under ideal simulation. They agree with the classical baseline's test predictions on 1500 of 1500 samples (a sanity check; no test labels are available to us).

### Noise mitigation (PS-Q3 required; optional for PS-Q1/Q2)
| Strategy | Mitigated metric | Extra circuit evals | Justified? |
|----------|-----------------|---------------------|-----------|
| Readout recalibration: re-fit only the 2 readout parameters (w, b) on noisy validation ⟨Z₀⟩, circuit weights fixed; scored out-of-fold (5-fold) | Submitted model 0.4731 → **0.9986**; direct ablation 0.7993 → 0.9612; 1-layer 0.9487 → 0.9640 | None beyond one noisy pass over the 1500 validation circuits (8192 shots each) | Yes, because noise mostly rescales ⟨Z₀⟩ (correlation 0.998 with the ideal value). It needs labelled calibration data and assumes the device noise stays the same. Reported as an analysis only; it is not used for the predictions |

---

## 3. Task A — Bell state |Φ+⟩ (required)

Produced by `task_a_bell()` in `benchmark/qff_qml_benchmark.py` (8192 shots per basis).

**Fidelity definition used:** F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩) / 4, with ⟨P⟩ = (N_even − N_odd)/N, SE(⟨P⟩) = √((1 − ⟨P⟩²)/N) and SE(F) = √(SE_XX² + SE_YY² + SE_ZZ²)/4. (P(00)+P(11) is a population, not a fidelity, and is not accepted.)

| Condition | ⟨XX⟩ ± SE | ⟨YY⟩ ± SE | ⟨ZZ⟩ ± SE | F ± SE |
|-----------|-----------|-----------|-----------|--------|
| Ideal | +1.0000 ± 0.0000 | −1.0000 ± 0.0000 | +1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Noisy (benchmark) | +0.8191 ± 0.0063 | −0.8279 ± 0.0062 | +0.8735 ± 0.0054 | 0.8801 ± 0.0026 |

*What the difference says about noise and entanglement:* In the ideal run all three correlators are exactly ±1, the signature of a pure |Φ+⟩. Depolarizing errors on the H and CX gates and 2% readout flips mix the state toward the maximally mixed state, so every correlator shrinks and F drops to 0.880. ZZ (0.874) degrades least because its circuit has no basis-change gates, while XX and YY pass through extra single-qubit gates (and S† for YY) before measurement. F remains far above 0.5, the value of a classical 50/50 mixture of |00⟩ and |11⟩, so the entanglement survives; but about 12% of the overlap is lost after a single CX, which foreshadows the much larger loss of our 16-CX classifier.

---

## 4. Processor D Benchmark (Task B)

Produced by `python benchmark/qff_qml_benchmark.py processor_d_config.json`.

**QFT-3 fidelity definition used:** mean over the 8 computational-basis inputs |j⟩ of F_j = ⟨ψ_j|ρ_j|ψ_j⟩, where ψ_j = QFT|j⟩ (ideal) and ρ_j is the density matrix of the 3 logical qubits after the transpiled circuit runs under the benchmark noise model (gate noise; no measurement). Ideal must be 1.0.

**Processor ID:** qff_team_ring4_chord_v1  **n_qubits:** 4  **Topology:** ring 0-1-2-3-0 plus chord 0-2, all links two-way; native gates rx, ry, rz, h, cx

| Benchmark | Value |
|-----------|-------|
| QFT-3 depth (before transpile) | 6 |
| QFT-3 depth (after transpile) | 16 |
| Two-qubit gate count | 9 |
| SWAP overhead | 0 |
| Fidelity (ideal) | 1.0 |
| Fidelity (noisy) | 0.677499 |
| Coupling map direction (one-way / two-way links) | two-way (all 5 links listed in both directions, 10 directed pairs) |

Design comparison (all measured with the official `task_b_qft3`):

| Design | Depth after | 2q gates | SWAPs | Noisy QFT-3 F |
|---|---|---|---|---|
| Linear 5, one-way links | 31 | 12 | 1 | 0.5504 |
| Linear 5, two-way links (organizer example) | 22 | 12 | 1 | 0.6036 |
| Ring 4, two-way | 19 | 12 | 1 | 0.6351 |
| Ring 4 + chord 0-2, two-way, no native h | 18 | 9 | 0 | 0.6684 |
| **Ring 4 + chord 0-2, two-way, native h (submitted)** | **16** | **9** | **0** | **0.6775** |

*What do the results tell you about your topology?* Direction matters: listing the same line one-way instead of two-way costs 9 extra layers of depth and 0.053 fidelity, because every reversed CX needs extra single-qubit gates. QFT-3 needs all three qubit pairs to interact, so a plain ring pays one SWAP (3 extra CX); the single chord 0-2 closes a triangle, removing the SWAP and reaching the 9-two-qubit-gate minimum, and native `h` trims depth from 18 to 16. For our own circuit the 4-qubit ring is exact: the 2-layer classifier keeps its 16 CX at depth 22 on Processor D versus 34 CX at depth 36 on the linear example, and since our noise analysis shows the classifier's accuracy falls with every CX, avoiding routing directly matters. The chord is one extra coupler (crosstalk and fabrication cost on real hardware) that the benchmark does not penalise; a second chord gave no further gain, so we stopped at one.

---

## 5. Resource Costs
| Resource | Value |
|----------|-------|
| Qubits used | 4 |
| Circuit depth | Logical 21 (16 CX); after `prepare_for_noisy_run()` 21 with 16 single-qubit gates and 16 CX; on Processor D 22 with 16 CX |
| Trainable parameters | 18 nominal (16 rotation angles + readout w, b); 12 effective (6 angles have structurally zero gradient) |
| Training time (approx) | 30 epochs: about 140 s (1 layer) and 210 s (2 layers) per run while 8 runs share a 10-core CPU; the full notebook runs in about 10 minutes |
| Total circuit evaluations | Training: 405,000 forward evaluations per run (375 batches × 16 samples + full train/validation scoring per epoch × 30 epochs), 6,480,000 for all 16 runs; gradients by backpropagation through the simulator, so no extra circuit executions. Noisy evaluation: 1,500 circuits × 8,192 shots × 5 seeds = 61.44 M shots per model |

---

## 6. Limitations and Honest Assessment

- **Quantum is not competitive with classical here.** Logistic regression on the 30 features is perfect on validation (and the organizers' baseline is 1.000 on the hidden test set), with no quantum hardware, no noise and fewer assumptions. Our classifier matches it only in ideal simulation (one validation error) and only after the angle-scaling fix. We claim no quantum advantage. This dataset is near-linearly separable, which a 4-qubit circuit can represent but has no reason to beat.
- **Noise is the real limitation.** Under QFF2026-v2 the submitted model falls to 0.466 macro-F1. The cause is understood: noise attenuates ⟨Z₀⟩ (0.40×) while preserving the ordering, and the decision threshold learned in ideal training no longer fits. A shallower circuit (1 layer, 8 CX) is far more robust (0.950) but weaker ideally and blind to PC2; recalibrating the readout on noisy validation data recovers 0.999. Any real deployment would need that kind of noise-aware calibration.
- **Selection bias.** Validation chose the epoch, configuration and encoding, and the scaled encoding was introduced after inspecting the direct encoding's behaviour; validation scores are therefore optimistic.
- **Ansatz redundancy.** The specified ansatz has dead parameters (all four RZ just before the final ring, RY on qubit 1 in the last layer, and RZ on qubit 2 in the first layer); a leaner ansatz could reach the same result with fewer gates.
- **Synthetic data.** Features are modelled on MIT-BIH statistics; no clinical validity is claimed.
- **Process note.** Early in the session, a shell row count (`wc -l`) over the q1 CSVs touched the test-features file; no content was read or used for modelling. The test features are read only in Section 10 of the notebook, for inference.

---

## 7. Reproducibility
```python
SEED = 42              # numpy, random, PCA, weight init and batch order, Aer simulator, transpiler
EXTRA_SEEDS = [43, 44, 45, 46]        # training-robustness reruns of the selected configurations
NOISY_SIM_SEEDS = [42, 43, 44, 45, 46]  # simulator seeds for the noisy evaluation
```
- Two full independent runs of `vqc_work/scripts/train_vqc.py` produced identical configs, 480-row training histories and resource tables (all non-timing fields).
- Retraining the submitted configuration reproduces its parameters exactly (max difference 0.0); reloading the saved JSON model reproduces validation probabilities to 5.6e-16.
- `python vqc_work/scripts/train_vqc.py --selftest` runs 12 checks: macro-F1 matches scikit-learn, BCE loss, gradients match finite differences, partial last batch, the CX-ring identity, exact Qiskit/PennyLane equivalence before and after transpilation, noised-basis coverage, bit ordering, readout noise, signal attenuation, and a guard that only the training/validation files are read.
- The noisy pipeline is verified in the notebook: X on qubit 0 gives ⟨Z₀⟩ = −1 and X on qubit 1 gives +1; an empty circuit reads 1 with probability 0.0181 (expected 0.02); executed circuits equal the trained circuit to 1e-15 before noise.

---

## 8. Optional Extensions
[ ] Second mitigation strategy (PS-Q3)
[x] Ablation studies: angle encoding (scaled vs direct), depth (1 vs 2 layers), 5 training seeds, 5 simulator seeds, structural (zero-gradient) analysis of the ansatz
[x] Other: readout recalibration under noise (Section 2), Processor D design comparison (Section 4)
