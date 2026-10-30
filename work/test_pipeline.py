"""Integration test for full PS-Q1 pipeline."""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.problem import (
    generate_dataset,
    preprocess_data,
    build_feature_map,
    build_ansatz,
    build_vqc_circuit,
    train_vqc,
    predict_vqc_ideal,
)
from src.processors import (
    load_processor_config,
    build_noise_model,
    get_simulator,
)
from src.routing import (
    transpile_for_processor,
    extract_circuit_metrics,
    build_transpilation_summary,
)
from src.metrics import (
    compute_classification_metrics,
    evaluate_noisy_circuit,
    build_metrics_summary,
    dataframe_to_markdown,
)
from src.visualization import (
    plot_coupling_map,
    plot_dataset_distribution,
    plot_training_loss,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_transpilation_comparison,
    plot_metrics_comparison,
    save_circuit_diagram_image,
)

print("Starting integration test...")

# 1. Dataset
X, y = generate_dataset(n_samples=200, n_features=4, n_informative=3, n_redundant=1, random_state=42, save_path="data/raw/dataset.csv")
data = preprocess_data(X, y, test_size=0.20, random_state=42, save_dir="data/processed")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
X_train_raw, X_test_raw = data["X_train_raw"], data["X_test_raw"]

print(f"Data prepared: Train {X_train.shape}, Test {X_test.shape}")

# Plot dataset distribution
plot_dataset_distribution(X, y, save_path="results/figures/dataset_distribution.png")

# 2. Classical Baseline (Logistic Regression)
lr = LogisticRegression(random_state=42)
lr.fit(X_train_raw, y_train)
lr_preds = lr.predict(X_test_raw)
lr_probs = lr.predict_proba(X_test_raw)[:, 1]
m_lr = compute_classification_metrics(y_test, lr_preds, lr_probs, model_name="Classical Baseline (Logistic Regression)")
print("Baseline LR:", m_lr)

# 3. Quantum Circuit Construction
fm, x_params = build_feature_map(n_qubits=4)
ansatz, theta_params = build_ansatz(n_qubits=4, reps=2)
vqc_logical = build_vqc_circuit(fm, ansatz, measure=False, name="VQC_Logical")
vqc_measured = build_vqc_circuit(fm, ansatz, measure=True, name="VQC_Measured")

save_circuit_diagram_image(vqc_logical, title="PS-Q1 Team-Authored VQC Architecture", save_path="results/figures/vqc_circuit.png")

# 4. Train VQC
opt_theta, loss_history = train_vqc(vqc_logical, x_params, theta_params, X_train, y_train, maxiter=80, random_state=42)
plot_training_loss(loss_history, save_path="results/figures/training_loss.png")
print(f"VQC trained: {len(loss_history)} evaluations, final loss: {loss_history[-1]:.4f}")

# 5. Evaluate Ideal Statevector VQC
vqc_ideal_exp, vqc_ideal_pred = predict_vqc_ideal(vqc_logical, x_params, theta_params, opt_theta, X_test)
vqc_ideal_prob = (vqc_ideal_exp + 1.0) / 2.0
m_ideal = compute_classification_metrics(y_test, vqc_ideal_pred, vqc_ideal_prob, model_name="Ideal VQC (Statevector)")
print("Ideal VQC:", m_ideal)

# 6. Load Processors & Transpile
cfg_A = load_processor_config("processors/processor_A.json")
cfg_B = load_processor_config("processors/processor_B.json")

plot_coupling_map(cfg_A["coupling_map"], cfg_A["num_qubits"], title="Processor A (5 Qubits, Linear)", save_path="results/figures/coupling_map_processor_A.png")
plot_coupling_map(cfg_B["coupling_map"], cfg_B["num_qubits"], title="Processor B (9 Qubits, Heavy-Hex Inspired)", save_path="results/figures/coupling_map_processor_B.png")

vqc_bound_measured = vqc_measured.assign_parameters({theta_params[i]: opt_theta[i] for i in range(len(theta_params))})
t_qcA = transpile_for_processor(vqc_bound_measured, cfg_A)
t_qcB = transpile_for_processor(vqc_bound_measured, cfg_B)

transpilation_df = build_transpilation_summary(vqc_measured, t_qcA, t_qcB)
transpilation_df.to_csv("results/tables/transpilation_comparison.csv", index=False)
with open("results/tables/transpilation_comparison.md", "w", encoding="utf-8") as f:
    f.write(dataframe_to_markdown(transpilation_df))
plot_transpilation_comparison(transpilation_df, save_path="results/figures/transpilation_metrics_bar.png")
print("Transpilation summary:\n", transpilation_df)

# 7. Noisy Simulation
sim_A = get_simulator(cfg_A, noisy=True)
sim_B = get_simulator(cfg_B, noisy=True)

exp_A, prob_A, pred_A = evaluate_noisy_circuit(t_qcA, x_params, X_test, sim_A, shots=cfg_A["shots"], seed_simulator=42)
exp_B, prob_B, pred_B = evaluate_noisy_circuit(t_qcB, x_params, X_test, sim_B, shots=cfg_B["shots"], seed_simulator=42)

m_A = compute_classification_metrics(y_test, pred_A, prob_A, model_name="Processor A (Noisy 5Q Linear)")
m_B = compute_classification_metrics(y_test, pred_B, prob_B, model_name="Processor B (Noisy 9Q Mesh)")
print("Processor A:", m_A)
print("Processor B:", m_B)

# 8. Comparison Summary & Plots
all_metrics = [m_lr, m_ideal, m_A, m_B]
summary_df = build_metrics_summary(all_metrics)
summary_df.to_csv("results/tables/classification_metrics.csv", index=False)
with open("results/tables/classification_metrics.md", "w", encoding="utf-8") as f:
    f.write(dataframe_to_markdown(summary_df))
print("Metrics Summary:\n", summary_df)

plot_metrics_comparison(summary_df, save_path="results/figures/classification_metrics_bar.png")

# Confusion matrices
cms_dict = {
    "Logistic Regression": m_lr["Confusion Matrix"],
    "Ideal VQC": m_ideal["Confusion Matrix"],
    "Processor A": m_A["Confusion Matrix"],
    "Processor B": m_B["Confusion Matrix"],
}
plot_confusion_matrices(cms_dict, save_path="results/figures/confusion_matrices.png")

# ROC curves
roc_dict = {}
for name, probs in [
    ("Classical Baseline (Logistic Regression)", lr_probs),
    ("Ideal VQC (Statevector)", vqc_ideal_prob),
    ("Processor A (Noisy 5Q Linear)", prob_A),
    ("Processor B (Noisy 9Q Mesh)", prob_B),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_dict[name] = (fpr, tpr, roc_auc_score(y_test, probs))

plot_roc_curves(roc_dict, save_path="results/figures/roc_curves.png")

print("Pipeline integration test successfully completed!")
