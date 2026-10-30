# Independent Audit Report: Phase 1 VQC Hardware Benchmark Equivalence

**Audit Scope**: Verification of Processor A (5-qubit linear) vs. Processor B (9-qubit mesh) classification equivalence  
**Audit Location**: `results/audit_run/`  
**Fixed Trained Weights**: $\vec{\theta} \in \mathbb{R}^{12}$ from seed 42 statevector training  
**Simulator Evaluation**: 20 independent AerSimulator seeds (42 to 61), 4096 shots/evaluation  

---

## Executive Summary & Root-Cause Diagnosis of Fresh Test-Set Accuracies

This independent audit investigates whether the identical **92.50% test accuracy** reported for Processor A and Processor B in Phase 1 represents genuine physical algorithmic equivalence, an artifact of pipeline reporting, or a limitation of discrete metric thresholding on a small sample size ($N_{\text{test}} = 40$).

### Key Audit Findings & Clarification on Fresh Data Generation:

1. **Why `make_classification(..., random_state=1234)` yielded ~20% (VQC) and 12.5% (Logistic Regression)**:
   - In Scikit-Learn, `make_classification` uses its `random_state` parameter to generate:
     1. The physical coordinates of cluster centroids placed on hypercube vertices,
     2. The internal linear covariance projection matrix $A$ for informative features,
     3. The linear combination coefficients $B$ used to construct the redundant feature,
     4. A random column permutation of the features (`generator.permutation(n_features)`),
     5. Random label assignments to clusters.
   - Calling `generate_dataset(..., random_state=1234)` generates an **entirely different synthetic classification problem** in a different coordinate basis with permuted feature columns.
   - When a fixed model (trained on seed 42) is evaluated on seed 1234 without retraining, it experiences **total distribution mismatch / concept drift**. This explains why **even Classical Logistic Regression collapses to 12.5%** and **Ideal noiseless VQC collapses to 20.0%**. It is an out-of-distribution evaluation issue, not a quantum compilation or execution flaw.

2. **In-Distribution Independent Evaluation (`train_test_split(..., random_state=1234)`)**:
   - When the same underlying distribution (seed 42) is split using a fresh seed (`random_state=1234`) to obtain an unseen, independent 40-sample test set:
     - **Classical Baseline (Logistic Regression)**: **$0.9750$ (97.5%)**
     - **Ideal VQC (Statevector)**: **$0.9750$ (97.5%)**
     - **Processor A (Noisy 5Q Linear)**: **$0.9750 \pm 0.0000$ (97.5%)** across all 20 seeds
     - **Processor B (Noisy 9Q Mesh)**: **$0.9750 \pm 0.0000$ (97.5%)** across all 20 seeds
   - This decisively confirms that the trained VQC generalizes strongly ($97.5\%$) to unseen data from the true distribution.

3. **The Probability Arrays are NOT Identical**:
   - Pointwise identity (`np.array_equal`) is **`False`** across all datasets and seeds.
   - On the Phase 1 test set: mean absolute difference $\overline{|p_A - p_B|} = 0.0198$, with maximum difference $0.0430$.
   - On the in-distribution independent test set: mean absolute difference $\overline{|p_A - p_B|} = 0.0210$, with maximum difference $0.0430$.

4. **Processor A Suffers Significant Hardware Signal Degradation**:
   - Processor A's linear topology requires **10 SWAP gates (+30 extra CX gates)**, giving **42 CX gates and depth 104** (a **3.50x CX overhead** and 2.6x depth expansion over Processor B's **12 CX gates and depth 40**).
   - This routing overhead attenuates the continuous classification signal: Processor A's distance from the decision boundary $|p - 0.5|$ is reduced by **15.6%** relative to Processor B.
   - Expectation MSE relative to ideal statevector is **$3.23	imes$ higher on Processor A** ($0.0097$ vs. $0.0030$).

---

## Direct Answers to Audit Objectives

### A. Are the identical 92.5% results reproducible?
**YES.**  
Re-running the pipeline on the original Phase 1 test set ($N=40$) under `seed_simulator=42` yields 37/40 ($92.50\%$) for both processors. Across 20 simulator seeds:
- **Processor A**: Accuracy = **$0.9263 \pm 0.0054$** (19 of 20 runs yielded $0.9250$; 1 run yielded $0.9500$).
- **Processor B**: Accuracy = **$0.9238 \pm 0.0054$** (19 of 20 runs yielded $0.9250$; 1 run yielded $0.9000$).
- **Ideal VQC (Statevector)**: Accuracy = **$0.9250$** (deterministic).
- **Classical Logistic Regression**: Accuracy = **$0.9250$** (deterministic).

---

### B. Are the raw probabilities different between Processor A and B?
**YES, unequivocally.**  
- Original benchmark set: mean absolute difference $\overline{|p_A - p_B|} = 0.01981$, $\max |p_A - p_B| = 0.04299$.
- In-distribution independent set: mean absolute difference $\overline{|p_A - p_B|} = 0.02105$, $\max |p_A - p_B| = 0.04299$.
- For positive samples ($y=1$), Processor B consistently produces higher confidence ($p_B > p_A$). For negative samples ($y=0$), Processor B consistently produces lower probability ($p_B < p_A$). Processor A's probabilities are systematically compressed toward $0.5$ by depolarizing noise.

---

### C. Does Processor A show greater noise sensitivity or variance?
**YES, at the continuous physical expectation and fidelity level.**
- **Physical Expectation Degradation**: Mean squared error relative to ideal statevector expectation $\langle ZZZZ angle_{\text{ideal}}$ is:
  $$\text{MSE}_{\text{exp}}(\text{Processor A}) = 0.0097 \pm 0.0004$$
  $$\text{MSE}_{\text{exp}}(\text{Processor B}) = 0.0030 \pm 0.0003$$
  Processor A incurs **$3.23\times$ higher expectation error** than Processor B.
- **Probability Calibration**: Brier score on the original test set shows Processor B's superior calibration ($0.1606 \pm 0.0010$ vs. $0.1716 \pm 0.0009$).

---

### D. Is the original result likely valid, inconclusive because of the small test set, or caused by a bug?
**The result is VALID and free of software bugs, but INCONCLUSIVE regarding physical processor equivalence due to metric coarseness on a small test set ($N=40$).**
- No-noise sanity check (`noisy=False`) confirms both transpiled circuits are unitary representations of the same algorithm, matching ideal statevector within shot-noise limits ($\max |\Delta| < 0.036$).
- The 3 samples misclassified by both processors (indices 10, 11, 36) are inherently hard samples misclassified by all models.
- For all other 37 samples, the margin from 0.5 was larger than the $\sim 0.02 - 0.04$ depolarizing shift caused by Processor A's routing overhead. Converting continuous probabilities into discrete 0/1 predictions completely masked the underlying fidelity degradation.

---

### E. What exact conclusion is safe to include in the final submission?

> **Scientifically Rigorous Conclusion for Submission:**
> 1. **Topological Compilation Advantage**: Processor B's 9-qubit mesh natively embeds the circular entangling graph of the 4-qubit VQC, achieving optimal compilation with **0 SWAP gates, 12 CNOT gates, and depth 40**. Processor A's linear topology requires **10 SWAP gates (+30 extra CNOTs, 3.50x CX overhead) and depth 104**.
> 2. **Physical State Degradation**: Under realistic gate depolarizing ($p_{\text{cx}}=0.005$) and readout ($p_{\text{ro}}=0.01$) noise, Processor A's routing overhead causes severe signal attenuation, increasing expectation error relative to ideal statevector simulation by **$3.23\times$** compared to Processor B ($0.0097$ vs. $0.0030$ MSE).
> 3. **Discrete Metric Masking**: On a small test set ($N=40$), this physical degradation is masked by discrete 0/1 decision thresholding, producing identical nominal classification accuracy ($92.50\%$ on Phase 1 split, $97.50\%$ on independent in-distribution split). Evaluating continuous probabilities, Brier scores, and multi-seed sweeps is essential to expose hardware topology bottlenecks on NISQ devices.
> 4. **What NOT to claim**: Do **not** claim that Processor B achieves higher test classification accuracy on the 40-sample benchmark, as discrete accuracy between the two architectures is statistically indistinguishable ($0.926 \pm 0.005$ vs. $0.924 \pm 0.005$). The demonstrable advantage of Processor B is in **compilation efficiency, circuit depth, and physical state fidelity**.

---

## Detailed Benchmark Metrics Table Across All Suites

| Dataset | Model | Accuracy_Mean | Accuracy_Std | Balanced_Acc_Mean | Balanced_Acc_Std | Macro_F1_Mean | Macro_F1_Std | Precision_Mean | Precision_Std | Recall_Mean | Recall_Std | ROC_AUC_Mean | ROC_AUC_Std | Brier_MSE_Mean | Brier_MSE_Std | Exp_Error_vs_Ideal_Mean | Exp_Error_vs_Ideal_Std | Confusion_Matrix_Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original Phase 1 (Seed 42) | Classical Baseline (Logistic Regression) | 0.925 | 0.0 | 0.9211 | 0.0 | 0.9238 | 0.0 | 1.0 | 0.0 | 0.8421 | 0.0 | 0.9298 | 0.0 | 0.0701 | 0.0 | 0.0 | 0.0 | TN=21, FP=0, FN=3, TP=16 |
| Original Phase 1 (Seed 42) | Ideal VQC (Statevector) | 0.925 | 0.0 | 0.9211 | 0.0 | 0.9238 | 0.0 | 1.0 | 0.0 | 0.8421 | 0.0 | 0.9173 | 0.0 | 0.1487 | 0.0 | 0.0 | 0.0 | TN=21, FP=0, FN=3, TP=16 |
| Original Phase 1 (Seed 42) | Processor A (Noisy 5Q Linear) | 0.9263 | 0.0056 | 0.9224 | 0.0059 | 0.9251 | 0.0057 | 1.0 | 0.0 | 0.8447 | 0.0118 | 0.9182 | 0.0044 | 0.1716 | 0.0009 | 0.0097 | 0.0004 | TN=21, FP=0, FN=3, TP=16 |
| Original Phase 1 (Seed 42) | Processor B (Noisy 9Q Mesh) | 0.9238 | 0.0056 | 0.9199 | 0.0053 | 0.9226 | 0.0055 | 0.9971 | 0.0132 | 0.8421 | 0.0 | 0.917 | 0.0027 | 0.1606 | 0.001 | 0.003 | 0.0003 | TN=21, FP=0, FN=3, TP=16 |
| In-Distribution Independent Split (Seed 1234) | Classical Baseline (Logistic Regression) | 0.975 | 0.0 | 0.9722 | 0.0 | 0.9746 | 0.0 | 1.0 | 0.0 | 0.9444 | 0.0 | 0.947 | 0.0 | 0.0454 | 0.0 | 0.0 | 0.0 | TN=22, FP=0, FN=1, TP=17 |
| In-Distribution Independent Split (Seed 1234) | Ideal VQC (Statevector) | 0.975 | 0.0 | 0.9722 | 0.0 | 0.9746 | 0.0 | 1.0 | 0.0 | 0.9444 | 0.0 | 0.9495 | 0.0 | 0.1312 | 0.0 | 0.0 | 0.0 | TN=22, FP=0, FN=1, TP=17 |
| In-Distribution Independent Split (Seed 1234) | Processor A (Noisy 5Q Linear) | 0.975 | 0.0 | 0.9722 | 0.0 | 0.9746 | 0.0 | 1.0 | 0.0 | 0.9444 | 0.0 | 0.9503 | 0.0027 | 0.1591 | 0.0013 | 0.0099 | 0.0006 | TN=22, FP=0, FN=1, TP=17 |
| In-Distribution Independent Split (Seed 1234) | Processor B (Noisy 9Q Mesh) | 0.975 | 0.0 | 0.9722 | 0.0 | 0.9746 | 0.0 | 1.0 | 0.0 | 0.9444 | 0.0 | 0.9507 | 0.0024 | 0.1454 | 0.0012 | 0.003 | 0.0003 | TN=22, FP=0, FN=1, TP=17 |
| Out-of-Distribution Generation (Seed 1234) | Classical Baseline (Logistic Regression) | 0.125 | 0.0 | 0.1317 | 0.0 | 0.1245 | 0.0 | 0.125 | 0.0 | 0.087 | 0.0 | 0.0921 | 0.0 | 0.7479 | 0.0 | 0.0 | 0.0 | TN=3, FP=14, FN=21, TP=2 |
| Out-of-Distribution Generation (Seed 1234) | Ideal VQC (Statevector) | 0.2 | 0.0 | 0.1969 | 0.0 | 0.198 | 0.0 | 0.2632 | 0.0 | 0.2174 | 0.0 | 0.1458 | 0.0 | 0.3544 | 0.0 | 0.0 | 0.0 | TN=3, FP=14, FN=18, TP=5 |
| Out-of-Distribution Generation (Seed 1234) | Processor A (Noisy 5Q Linear) | 0.2125 | 0.019 | 0.2047 | 0.0185 | 0.2067 | 0.0183 | 0.2899 | 0.0242 | 0.2565 | 0.0312 | 0.148 | 0.0091 | 0.321 | 0.0013 | 0.0075 | 0.0003 | TN=3, FP=14, FN=17, TP=6 |
| Out-of-Distribution Generation (Seed 1234) | Processor B (Noisy 9Q Mesh) | 0.2075 | 0.02 | 0.2008 | 0.0202 | 0.2028 | 0.0199 | 0.2821 | 0.0227 | 0.2457 | 0.0255 | 0.1489 | 0.0058 | 0.3365 | 0.0013 | 0.0022 | 0.0002 | TN=3, FP=14, FN=17, TP=6 |

---

## Transpilation Architecture Verification

| Architecture | Total Qubits | Circuit Depth | CX Gate Count | Extra CX (Routing) | Estimated SWAPs | CX Overhead Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logical (Ideal)** | 4 | 19 | 12 | 0 | 0 | $1.00\times$ |
| **Processor B (9Q Mesh)** | 9 | 40 | 12 | 0 | 0 | $1.00\times$ |
| **Processor A (5Q Linear)** | 5 | 104 | 42 | +30 | 10 | $3.50\times$ |

---

## Artifact Index
- **Audit Metrics**: `results/audit_run/audit_metrics.csv`
- **Per-Sample Predictions**: `results/audit_run/per_sample_predictions.csv`
- **Seed-by-Seed Metrics**: `results/audit_run/audit_metrics_per_seed.csv`
- **Comparison Visualization**: `results/audit_run/comparison_plot.png`
