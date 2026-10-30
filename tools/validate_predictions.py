"""
Check a predictions file before you submit it.

    python tools/validate_predictions.py q1 predictions_q1.csv
    python tools/validate_predictions.py q3 predictions_q3.csv

Rules (kit v3):
  * Columns: sample_id,label            for q1 and q2
             sample_id,label,score      for q3
  * One row per test sample, ids exactly s0, s1, ... as in the sample_id column
    of the test-features file (q1: 1500 rows, q2: 200, q3: 500). Any row order.
  * label: 0/1 for q1 and q3, A/B for q2.
  * score (q3 only): your model's probability that label = 1, a number in [0, 1].
    ROC-AUC is computed from score; F1 is computed from label.
"""
import sys

import pandas as pd

SPEC = {"q1": (1500, ["sample_id", "label"], {"0", "1"}),
        "q2": (200, ["sample_id", "label"], {"A", "B"}),
        "q3": (500, ["sample_id", "label", "score"], {"0", "1"})}


def validate(ps: str, path: str) -> list:
    n, cols, labels = SPEC[ps]
    errs = []
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:  # noqa: BLE001
        return [f"cannot read CSV: {e}"]
    if list(df.columns) != cols:
        return [f"columns must be exactly {cols}, found {list(df.columns)}"]
    if len(df) != n:
        errs.append(f"expected {n} rows, found {len(df)}")
    expected = {f"s{i}" for i in range(n)}
    ids = set(df["sample_id"])
    if df["sample_id"].duplicated().any():
        errs.append("duplicate sample_id values")
    if ids != expected:
        errs.append(f"sample_id must be s0..s{n - 1}; missing {len(expected - ids)}, unexpected {len(ids - expected)}")
    bad = sorted(set(df["label"].str.strip()) - labels)
    if bad:
        errs.append(f"invalid label values {bad[:5]}; allowed {sorted(labels)}")
    if ps == "q3":
        s = pd.to_numeric(df["score"], errors="coerce")
        if s.isna().any() or (s < 0).any() or (s > 1).any():
            errs.append("score must be a number in [0, 1] on every row")
    return errs


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in SPEC:
        print(__doc__)
        sys.exit(1)
    problems = validate(sys.argv[1], sys.argv[2])
    if problems:
        print("INVALID:")
        for p in problems:
            print("  -", p)
        sys.exit(2)
    print("OK - predictions file is valid.")
