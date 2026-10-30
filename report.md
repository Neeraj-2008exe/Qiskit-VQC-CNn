# PS-Q1 Architectural Report: Variational Quantum Classification & Hardware Topology Benchmarking

**Event / Track**: Qiskit Fall Fest 2026 — Phase 1 Submission  
**Task**: PS-Q1 Variational Quantum Classifier  
**Status**: Completed and Verified via `main.ipynb` Execution  

---

## 1. Problem and Objective

The primary objective of the PS-Q1 project is to investigate the architectural and algorithmic trade-offs of mapping a team-authored **4-qubit Variational Quantum Classifier (VQC)** onto two distinct quantum hardware topologies:
- **Processor A**: A 5-qubit 1-dimensional linear chain topology (`0 - 1 - 2 - 3 - 4`).
- **Processor B**: A 9-qubit 2-dimensional sparse heavy-hex-inspired mesh topology containing cyclic subgraphs.

We evaluate how hardware connectivity impacts:
1. Physical circuit transpilation (circuit depth, total operations, physical CNOT count).
2. Routing and SWAP overhead required to realize entangling layers.
3. Classification performance (Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrices) under realistic gate depolarizing and readout errors compared to an ideal statevector simulation and a classical Logistic Regression baseline.

---

## 2. Dataset and Preprocessing

The dataset is synthetically generated via Scikit-Learn according to the PS-Q1 specification:
- **Sample size**: $N = 200$ samples.
- **Dimensionality**: $n_{\text{features}} = 4$ ($3$ informative features, $1$ redundant feature formed by linear combination).
- **Class labels**: Binary target $y \in \{0, 1\}$.
- **Random seed**: 42.

### Train/Test Partitioning
- **Split ratio**: 80% training ($N_{\text{train}} = 160$), 20% testing ($N_{\text{test}} = 40$).
- **Stratification / Class balance**:
  - Training set: 78 samples of Class 0, 82 samples of Class 1.
  - Test set: 21 samples of Class 0, 19 samples of Class 1.

### Feature Scaling
To encode real-valued continuous features into quantum rotation angles without phase ambiguity, the features are scaled into the interval $[0, \pi]$ using `MinMaxScaler` fit strictly on the training partition:
$$\tilde{x}_j = \pi \cdot \frac{x_j - \min(X_{\text{train}, j})}{\max(X_{\text{train}, j}) - \min(X_{\text{train}, j})}$$
Raw data is persisted at `data/raw/dataset.csv`, and processed train/test partitions at `data/processed/train.csv` and `data/processed/test.csv`.

---

## 3. VQC Architecture and Training

We designed a modular, team-authored 4-qubit variational circuit that avoids deprecated library wrappers and provides explicit parameter control:

### 3.1 Feature Map ($U_{\Phi}(x)$)
- **Superposition**: Hadamard layer $H^{\otimes 4}$ preparing $|+\rangle^{\otimes 4}$.
- **Rotational Encoding**: Parameterized $R_z(x_i)$ gates for each qubit $i \in \{0, 1, 2, 3\}$.
- **Feature Entanglement**: Circular CNOT ring: $(0 \to 1, 1 \to 2, 2 \to 3, 3 \to 0)$ to capture non-linear pairwise interactions.
- **Second Rotation Layer**: Parameterized $R_z(x_i)$ gates.

### 3.2 Variational Ansatz ($W(\theta)$)
- **Parameterized Layers**: 2 repetitions of single-qubit $R_y(\theta_j)$ rotations followed by circular CNOT entanglement $(0 \to 1, 1 \to 2, 2 \to 3, 3 \to 0)$.
- **Final Readout Layer**: $R_y(\theta_k)$ rotation on each qubit.
- **Trainable Parameters**: $(2 + 1) \times 4 = 12$ variational angles $\vec{\theta} \in [-\pi, \pi]^{12}$.

### 3.3 Classification Observable and Loss Function
- **Observable**: 4-qubit parity operator $\mathcal{O} = Z_0 Z_1 Z_2 Z_3$.
- **Model Output**: $\hat{y}_{\text{cont}}(x, \vec{\theta}) = \langle \psi(x, \vec{\theta}) | ZZZZ | \psi(x, \vec{\theta}) \rangle \in [-1, 1]$.
- **Decision Rule**: $\hat{y} = 1$ if $\hat{y}_{\text{cont}} \ge 0$, else $0$.
- **Loss Function**: Mean Squared Error (MSE) evaluated against target labels mapped to $y_i^{\pm} \in \{-1, +1\}$:
  $$\mathcal{L}(\vec{\theta}) = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \left( \hat{y}_{\text{cont}}(x_i, \vec{\theta}) - y_i^{\pm} \right)^2$$

### 3.4 Optimization Dynamics
- **Optimizer**: COBYLA (constrained optimization by linear approximation), max 80 iterations.
- **Computational Acceleration**: Input features are pre-bound across the 160 training circuits, reducing per-iteration cost to ~180 ms.
- **Convergence**: Initial MSE loss $= 1.4168$; final optimized loss $= 0.6054$ reached in 81 evaluations.

---

## 4. Classical Baseline

A classical linear classifier baseline was established using Scikit-Learn `LogisticRegression(random_state=42)` fit on the unscaled training data:
- **Test Accuracy**: $0.9250$ (37 correct out of 40)
- **Balanced Accuracy**: $0.9211$
- **Precision**: $1.0000$ (0 false positives)
- **Recall**: $0.8421$ (16 true positives, 3 false negatives)
- **F1 Score**: $0.9143$
- **ROC-AUC**: $0.9298$
- **Confusion Matrix**:
  $$\begin{pmatrix} 21 & 0 \\ 3 & 16 \end{pmatrix}$$

---

## 5. Processor A Definition and Results

- **Configuration Source**: `processors/processor_A.json`
- **Qubits**: 5 qubits.
- **Coupling Map**: Linear chain: $[(0, 1), (1, 2), (2, 3), (3, 4)]$.
- **Native Basis Gates**: `['rz', 'sx', 'x', 'cx']`.
- **Noise Specification**:
  - Single-qubit depolarizing error: $p_{1q} = 0.001$ on `sx`, `x`.
  - Two-qubit depolarizing error: $p_{\text{cx}} = 0.005$ on `cx`.
  - Readout error: $p_{\text{ro}} = 0.01$ bit-flip on measurement.
- **Transpilation Settings**: `optimization_level = 1`, `seed_transpiler = 42`.
- **Physical Transpiled Circuit**:
  - Total Qubits: 5
  - Physical Depth: 104 (Logical depth: 19; Depth overhead: $+85$)
  - Total Operations: 264
  - Physical CX Count: 42 (Logical CX: 12; Extra routing CX: $+30$)
  - Estimated SWAP Gates: 10
  - CX Overhead Ratio: $3.500\times$ (+250% overhead)
- **Noisy Physical Execution Results (4096 shots)**:
  - Test Accuracy: $0.9250$
  - Balanced Accuracy: $0.9211$
  - Precision: $1.0000$
  - Recall: $0.8421$
  - F1 Score: $0.9143$
  - ROC-AUC: $0.9148$

---

## 6. Processor B Definition and Results

- **Configuration Source**: `processors/processor_B.json`
- **Qubits**: 9 qubits.
- **Coupling Map**: Heavy-hex-inspired mesh with 12 bidirectional coupling links:
  $[(0, 1), (1, 2), (1, 6), (2, 3), (2, 6), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (8, 0)]$.
- **Native Basis Gates**: `['rz', 'sx', 'x', 'cx']`.
- **Noise Specification**: Identical to Processor A ($p_{1q} = 0.001$, $p_{\text{cx}} = 0.005$, $p_{\text{ro}} = 0.01$).
- **Transpilation Settings**: `optimization_level = 1`, `seed_transpiler = 42`.
- **Physical Transpiled Circuit**:
  - Total Qubits: 9
  - Physical Depth: 40 (Logical depth: 19; Depth overhead: $+21$)
  - Total Operations: 112
  - Physical CX Count: 12 (Logical CX: 12; Extra routing CX: $0$)
  - Estimated SWAP Gates: 0
  - CX Overhead Ratio: $1.000\times$ (0% overhead)
- **Noisy Physical Execution Results (4096 shots)**:
  - Test Accuracy: $0.9250$
  - Balanced Accuracy: $0.9211$
  - Precision: $1.0000$
  - Recall: $0.8421$
  - F1 Score: $0.9143$
  - ROC-AUC: $0.9148$

---

## 7. A/B Comparison: Depth, CX Count, SWAP Overhead, and Metrics

### 7.1 Transpilation & Routing Overhead Comparison

| Architecture | Total Qubits | Circuit Depth | Total Ops | CX Count | Extra CX | CX Ratio | Est. SWAPs | Depth Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logical (Ideal)** | 4 | 19 | 40 | 12 | 0 | $1.00\times$ | 0 | 0 |
| **Processor A (5Q Linear)** | 5 | 104 | 264 | 42 | 30 | $3.50\times$ | 10 | +85 |
| **Processor B (9Q Mesh)** | 9 | 40 | 112 | 12 | 0 | $1.00\times$ | 0 | +21 |

### Key Architectural Finding:
The circular entangling connectivity of the logical algorithm matches a 4-qubit cyclic subgraph present natively within Processor B's heavy-hex topology (e.g., cycle $0-1-6-8-0$ or $1-2-5-6-1$). As a result:
- **Processor B** required **zero SWAP routing gates** and **zero extra CNOT gates**, preserving the minimal 12 CX count and maintaining depth at 40.
- **Processor A**, constrained by a 1D linear chain, could not directly execute non-adjacent two-qubit interactions (specifically between the terminal qubits of the ring). Routing required **10 SWAP gates (30 extra CNOT gates)**, causing a **3.5x explosion in two-qubit gate count** and a **5.5x increase in depth**.

### 7.2 Classification Performance Comparison

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | Correct / 40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Logistic Regression** | 0.9250 | 0.9211 | 1.0000 | 0.8421 | 0.9143 | 0.9298 | 37 / 40 |
| **Ideal VQC (Statevector)** | 0.9250 | 0.9211 | 1.0000 | 0.8421 | 0.9143 | 0.9173 | 37 / 40 |
| **Processor A (Noisy 5Q)** | 0.9250 | 0.9211 | 1.0000 | 0.8421 | 0.9143 | 0.9148 | 37 / 40 |
| **Processor B (Noisy 9Q)** | 0.9250 | 0.9211 | 1.0000 | 0.8421 | 0.9143 | 0.9148 | 37 / 40 |

All models produced an identical confusion matrix on the 40 test samples:
- True Negatives (Class 0): 21
- False Positives (Class 0 predicted as 1): 0
- False Negatives (Class 1 predicted as 0): 3
- True Positives (Class 1): 16

---

## 8. Limitations and Threats to Validity

1. **Synthetic Problem Scale**:
   The current benchmark uses a synthetic 4-feature dataset with 200 samples. While well-suited for demonstrating topological routing overhead, real-world utility benchmarks require higher dimensional datasets where bar-ren plateaus and trainability challenges emerge.
2. **Noise Model Assumptions**:
   The noise model assumes uniform depolarizing gate errors ($p_{1q}=0.001$, $p_{\text{cx}}=0.005$) and uniform readout errors ($p_{\text{ro}}=0.01$) across all physical qubits. On real hardware, gate fidelities and $T_1/T_2$ coherence times vary substantially per qubit and link, which would further penalize the longer depth of Processor A.
3. **Optimized Logical Mapping**:
   The circuit feature map uses circular entanglement, which directly stresses 1D linear processors. Linear or nearest-neighbor logical circuits would incur less SWAP overhead on Processor A, though at the expense of feature entanglement expressiveness.

---

## 9. Reproducibility

Every component of this benchmark is deterministically reproducible:
- **Environment**: Python 3.13.16 with `qiskit==2.6.0`, `qiskit-aer==0.17.2`, `scikit-learn==1.9.1`.
- **Random Seeds**:
  - Dataset generation seed: `42`
  - Train/test split seed: `42`
  - COBYLA parameter initialization seed: `42`
  - Transpiler seed: `42` (`seed_transpiler=42`)
  - Aer simulation seed: `42` (`seed_simulator=42`)
- **Execution Script**:
  Run `main.ipynb` sequentially from top to bottom. All intermediate and final data are saved in `data/` and `results/`.

---

## 10. Conclusion

This controlled simulation demonstrates the role of quantum processor topology in circuit compilation:
1. **Algorithmic Parity**: The team-authored 4-qubit VQC matches the classical Logistic Regression baseline (0.925 accuracy, 0.9143 F1 score).
2. **Topological Advantage**: Processor B's cyclic connectivity completely eliminated routing overhead (0 SWAPs, 0 extra CX gates), whereas Processor A's linear topology incurred 10 SWAPs (+30 CX gates, +250% two-qubit overhead) and increased circuit depth by +85 steps.
3. **Hardware-Algorithm Co-design**: Choosing a physical hardware architecture whose coupling graph naturally reflects the entangling structure of the quantum algorithm is paramount to minimizing error accumulation on NISQ devices.
