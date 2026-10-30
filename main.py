"""PS-Q1 Variational Quantum Classifier end-to-end experiment runner.

Executes the synthetic 4-qubit VQC experiment, classical Logistic Regression baseline,
transpilation for Processor A (5-qubit linear) and Processor B (9-qubit mesh),
and exports metrics and comparison tables.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.problem import (
    generate_dataset,
    preprocess_data,
    build_feature_map,
    build_ansatz,
    build_vqc_circuit,
    train_vqc,
    predict_vqc_ideal,
)
from src.processors import load_processor_config, get_simulator
from src.routing import transpile_for_processor, build_transpilation_summary
from src.metrics import (
    compute_classification_metrics,
    evaluate_noisy_circuit,
    build_metrics_summary,
    dataframe_to_markdown,
)

ROOT = Path(__file__).parent


def main():
    print("=== Running PS-Q1 End-to-End Experiment ===")

    # 1. Dataset Generation and Preprocessing
    X, y = generate_dataset(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        random_state=42,
        save_path=str(ROOT / "data" / "raw" / "dataset.csv"),
    )
    data = preprocess_data(
        X,
        y,
        test_size=0.20,
        random_state=42,
        save_dir=str(ROOT / "data" / "processed"),
    )
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    X_train_raw, X_test_raw = data["X_train_raw"], data["X_test_raw"]

    # 2. Classical Logistic Regression Baseline
    lr = LogisticRegression(random_state=42).fit(X_train_raw, y_train)
    lr_pred = lr.predict(X_test_raw)
    lr_prob = lr.predict_proba(X_test_raw)[:, 1]
    m_lr = compute_classification_metrics(
        y_test, lr_pred, lr_prob, model_name="Classical Baseline (Logistic Regression)"
    )

    # 3. Quantum Circuit Construction & Training
    fm, x_params = build_feature_map(4)
    ansatz, theta_params = build_ansatz(4, reps=2)
    vqc_log = build_vqc_circuit(fm, ansatz, measure=False)
    vqc_meas = build_vqc_circuit(fm, ansatz, measure=True)

    opt_theta, loss_history = train_vqc(
        vqc_log, x_params, theta_params, X_train, y_train, maxiter=80, random_state=42
    )
    vqc_exp, vqc_pred = predict_vqc_ideal(
        vqc_log, x_params, theta_params, opt_theta, X_test
    )
    m_ideal = compute_classification_metrics(
        y_test, vqc_pred, (vqc_exp + 1.0) / 2.0, model_name="Ideal VQC (Statevector)"
    )

    # 4. Processor Loading & Transpilation Analysis
    cfg_A = load_processor_config(str(ROOT / "processors" / "processor_A.json"))
    cfg_B = load_processor_config(str(ROOT / "processors" / "processor_B.json"))

    vqc_bound = vqc_meas.assign_parameters(
        {theta_params[i]: opt_theta[i] for i in range(len(theta_params))}
    )
    t_A = transpile_for_processor(vqc_bound, cfg_A)
    t_B = transpile_for_processor(vqc_bound, cfg_B)

    trans_df = build_transpilation_summary(vqc_meas, t_A, t_B)
    trans_df.to_csv(ROOT / "results" / "tables" / "transpilation_comparison.csv", index=False)
    with open(ROOT / "results" / "tables" / "transpilation_comparison.md", "w", encoding="utf-8") as f:
        f.write(dataframe_to_markdown(trans_df))

    # 5. Noisy Aer Execution
    sim_A = get_simulator(cfg_A, noisy=True)
    sim_B = get_simulator(cfg_B, noisy=True)
    _, prob_A, pred_A = evaluate_noisy_circuit(
        t_A, x_params, X_test, sim_A, shots=cfg_A["shots"], seed_simulator=42
    )
    _, prob_B, pred_B = evaluate_noisy_circuit(
        t_B, x_params, X_test, sim_B, shots=cfg_B["shots"], seed_simulator=42
    )

    m_A = compute_classification_metrics(
        y_test, pred_A, prob_A, model_name="Processor A (Noisy 5Q Linear)"
    )
    m_B = compute_classification_metrics(
        y_test, pred_B, prob_B, model_name="Processor B (Noisy 9Q Mesh)"
    )

    # 6. Save Benchmark Tables
    summary_df = build_metrics_summary([m_lr, m_ideal, m_A, m_B])
    summary_df.to_csv(ROOT / "results" / "tables" / "classification_metrics.csv", index=False)
    with open(ROOT / "results" / "tables" / "classification_metrics.md", "w", encoding="utf-8") as f:
        f.write(dataframe_to_markdown(summary_df))

    benchmark_rows = [
        {
            "processor": "Processor A",
            "accuracy": m_A["Accuracy"],
            "f1": m_A["F1 Score"],
            "recall": m_A["Recall"],
            "depth": int(trans_df.loc[trans_df["Name"].str.contains("Processor A"), "Depth"].values[0]),
            "cx_count": int(trans_df.loc[trans_df["Name"].str.contains("Processor A"), "CX Count"].values[0]),
            "swap_count": int(trans_df.loc[trans_df["Name"].str.contains("Processor A"), "Estimated SWAPs"].values[0]),
        },
        {
            "processor": "Processor B",
            "accuracy": m_B["Accuracy"],
            "f1": m_B["F1 Score"],
            "recall": m_B["Recall"],
            "depth": int(trans_df.loc[trans_df["Name"].str.contains("Processor B"), "Depth"].values[0]),
            "cx_count": int(trans_df.loc[trans_df["Name"].str.contains("Processor B"), "CX Count"].values[0]),
            "swap_count": int(trans_df.loc[trans_df["Name"].str.contains("Processor B"), "Estimated SWAPs"].values[0]),
        },
    ]
    pd.DataFrame(benchmark_rows).to_csv(ROOT / "results" / "tables" / "benchmark.csv", index=False)

    model_metrics_rows = [
        {"model": "VQC", "accuracy": m_ideal["Accuracy"], "f1": m_ideal["F1 Score"], "recall": m_ideal["Recall"]},
        {"model": "LogisticRegression", "accuracy": m_lr["Accuracy"], "f1": m_lr["F1 Score"], "recall": m_lr["Recall"]},
    ]
    pd.DataFrame(model_metrics_rows).to_csv(ROOT / "results" / "tables" / "model_metrics.csv", index=False)

    print("\n=== Transpilation Summary ===")
    print(trans_df.to_string(index=False))
    print("\n=== Classification Summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== PS-Q1 Execution Finished Successfully ===")


if __name__ == "__main__":
    main()
