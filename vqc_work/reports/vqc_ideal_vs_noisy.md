# PS-Q1 VQC - ideal vs official-noise validation

The *same* trained weights are evaluated twice on the 1,500 validation rows: exact statevector (ideal) and the official QFF2026-v2 noise model (noisy). Nothing is retrained or tuned on noisy results. Validation-only; not held-out test evidence.

## Official noise configuration
- Helper `noise_model/benchmark_noise_model.py`, imported unchanged; version **QFF2026-v2**: single-qubit depolarizing 0.01, two-qubit depolarizing 0.05, symmetric readout error 0.02; simulator `noisy_simulator(seed)`.
- **Transpilation: `prepare_for_noisy_run()` is applied to every executed circuit** (each validation sample's bound circuit; basis rx, ry, rz, cx; optimisation level 1; `seed_transpiler=42`); the helper itself asserts that every gate carries noise.
- Shots: 8192 per circuit (helper default and Task A's value; no PS-Q1 value is documented). Simulator seeds [42, 43, 44, 45, 46]; the seed-42 run is the primary number and the five-seed mean +- sd shows shot-noise variability. `<Z0> = (n0 - n1)/shots` from one classical bit mapped from qubit 0.

## Verification evidence (canonical model)
| check | result |
|---|---|
| Noise model attached to simulator | True (noise instructions: cx, cz, h, measure, rx, ry, rz, s, sdg, swap, sx, t, tdg, x, y, z) |
| Official parameters | p1q=0.01, p2q=0.05, readout=0.02, version QFF2026-v2 |
| Bit order: X on q0 -> <Z0> | -1.000 (expected -1): ok |
| Bit order: X on q1 -> <Z0> | +1.000 (expected +1): ok |
| Readout noise on an empty circuit, P(1) | 0.0181 vs 0.02 (SE 0.0015): ok |
| Qiskit logical circuit vs PennyLane (exact statevector, 300 val samples) | max |diff| 1.2e-15: ok |
| Officially transpiled circuit vs PennyLane (exact statevector) | max |diff| 1.1e-15: ok |
| Noiseless shots vs exact | max |diff| 0.0284 (shot sigma 0.0110): ok |
| Noisy vs exact on the same subset | mean |z| 0.5389 -> 0.2134 |

Overall: **all checks passed**.

## Ideal vs noisy macro-F1
| model | CX | ideal | noisy (seed 42) | noisy mean +- sd (5 sim seeds) | change | noisy recall 0 | noisy recall 1 | ROC-AUC ideal | ROC-AUC noisy | <Z0> attenuation slope | corr(ideal, noisy <Z0>) | decision threshold in <Z0> | flipped predictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| canonical: scaled, 2 layer(s) | 16 | 0.9993 | 0.4731 | 0.4663 +- 0.0048 | -0.5262 | 0.1924 | 1.0000 | 1.0000 | 1.0000 | 0.396 | 0.9977 | -0.375 | 0.483 |
| ablation: direct, 2 layer(s) | 16 | 0.9633 | 0.7993 | 0.7971 +- 0.0037 | -0.1640 | 0.6652 | 1.0000 | 0.9921 | 0.9911 | 0.380 | 0.9968 | -0.219 | 0.196 |
| diagnostic: scaled, 1 layer | 8 | 0.9667 | 0.9487 | 0.9496 +- 0.0013 | -0.0180 | 0.9255 | 0.9867 | 0.9940 | 0.9938 | 0.599 | 0.9996 | -0.196 | 0.035 |

Canonical model, ideal confusion matrix:

| true \ pred | 0 Normal | 1 Arrhythmia |
|---|---|---|
| 0 Normal | 898 | 1 |
| 1 Arrhythmia | 0 | 601 |

Canonical model, noisy confusion matrix (seed 42):

| true \ pred | 0 Normal | 1 Arrhythmia |
|---|---|---|
| 0 Normal | 173 | 726 |
| 1 Arrhythmia | 0 | 601 |

Canonical per class, ideal:

| class | precision | recall | F1 |
|---|---|---|---|
| 0 Normal | 1.0000 | 0.9989 | 0.9994 |
| 1 Arrhythmia | 0.9983 | 1.0000 | 0.9992 |

Canonical per class, noisy (seed 42):

| class | precision | recall | F1 |
|---|---|---|---|
| 0 Normal | 1.0000 | 0.1924 | 0.3228 |
| 1 Arrhythmia | 0.4529 | 1.0000 | 0.6234 |

Predicted P(class 1), ideal:

| mean | std | min | p5 | p25 | p50 | p75 | p95 | max | frac predicted 1 |
|---|---|---|---|---|---|---|---|---|---|
| 0.3910 | 0.4406 | 0.0111 | 0.0136 | 0.0197 | 0.0430 | 0.9278 | 0.9681 | 0.9998 | 0.4013 |

Predicted P(class 1), noisy (seed 42):

| mean | std | min | p5 | p25 | p50 | p75 | p95 | max | frac predicted 1 |
|---|---|---|---|---|---|---|---|---|---|
| 0.7082 | 0.1947 | 0.4369 | 0.4835 | 0.5289 | 0.6116 | 0.9350 | 0.9526 | 0.9938 | 0.8847 |

## Why noise changes macro-F1
- **Attenuation, not scrambling.** Under QFF2026-v2 the measured `<Z0>` of the canonical model is close to a scaled copy of the ideal value (slope 0.396, correlation 0.9977), and the ranking of samples is largely preserved (ROC-AUC 1.0000 ideal vs 1.0000 noisy). Depolarizing errors on 16 CX gates (5% each), single-qubit gates (1%) and 2% readout flips shrink the expectation value toward zero.
- **Fixed decision rule.** The trained readout puts the decision boundary at `<Z0> = -0.375`. That threshold was learned for the unattenuated signal; after shrinkage many samples of one class cross it, so the damage is class-specific (noisy recall: class 0 0.192, class 1 1.000). The information largely survives; the calibration of the decision rule does not.
- **Depth trade-off.** The 1-layer scaled model (8 CX) keeps more signal under noise (attenuation slope 0.599, noisy macro-F1 0.9496) but has a lower ideal score (0.9667) and, as shown in the training report, cannot use PC2. The canonical 2-layer model is best ideally and most fragile under noise.
- The direct-angle ablation degrades from 0.9633 to 0.7971.

## Circuit resources under the official transpilation
| model | depth | 1q gates | CX | ops outside basis | un-noised gates | circuits | shots (5 seeds) | transpile time (s) | run time per seed (s) |
|---|---|---|---|---|---|---|---|---|---|
| canonical: scaled, 2 layer(s) | 21.0 (21-21) | 16.0 | 16-16 | none | none | 1500 | 61,440,000 | 2.6 | 8.8 |
| ablation: direct, 2 layer(s) | 21.0 (21-21) | 16.0 | 16-16 | none | none | 1500 | 61,440,000 | 2.6 | 8.8 |
| diagnostic: scaled, 1 layer | 11.0 (11-11) | 8.0 | 8-8 | none | none | 1500 | 61,440,000 | 2.4 | 8.1 |

Logical circuits: 1 layer depth 11 / CX 8; 2 layers depth 21 / CX 16. Transpilation merges the encoding `RY` with the first trainable `RY` on each qubit (fewer single-qubit gates than the logical circuit) and adds no CX. No coupling map is applied (basis only), so no SWAPs are included here; mapping onto Processor D is part of Task B.

## Correction made during the audit
Earlier versions transpiled one *parametrised* template and bound parameters afterwards. With symbolic parameters the transpiler cannot merge the encoding `RY` with the first trainable `RY`, so 4 extra noisy gates per circuit were executed and noisy macro-F1 was understated (direct 2-layer model: 0.778 with the template method vs 0.797 per-circuit, 5-seed means). The pipeline now runs the official `prepare_for_noisy_run()` on every executed circuit, as the rules state.

## Blockers
None.

## Caveats
- Noisy numbers include shot noise (SE of each `<Z0>` about 0.0110); variability over 5 simulator seeds is reported.
- Ideal numbers are exact, so ideal-vs-noisy differences mix device noise and finite-shot effects.
- The decision threshold is the one learned in ideal training; no noise-aware recalibration is applied in this workstream.
