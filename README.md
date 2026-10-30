# QFF2026 QML Track - Phase 2: PS-Q1

**Problem statement: PS-Q1 (ECG heartbeat classification, QML-BIO-MITBIH).** Common tasks: **Task A** (Bell state) and **Task B** (Processor D).
**Team:** Deuterium
**Members:** Charan Reddy, Krishna Neeraj, Ismail Mohammad

(Same team and problem statement as Phase 1.)

Everything runs from `solution.ipynb`: Task A, Task B, the classical baseline, the hybrid quantum classifier (training, ideal and noisy evaluation) and the test predictions.
The full report is [`results.md`](results.md).

## Setup (Python 3.11 - 3.13)

```bash
python3 -m venv qml-env
source qml-env/bin/activate            # Windows: qml-env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt        # the organizer's pinned kit v3 versions
```

## Run

All commands are run from the repository root.

```bash
python noise_model/benchmark_noise_model.py
```

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 solution.ipynb
```

The notebook takes about 10 minutes on a 10-core laptop (training runs in parallel processes; allow 15-30 minutes on fewer cores). It rewrites `predictions_q1.csv`. Opening it with `jupyter notebook solution.ipynb` and running all cells works too.

Check the deliverables:

```bash
python tools/validate_predictions.py q1 predictions_q1.csv
```

```bash
python benchmark/qff_qml_benchmark.py processor_d_config.json
```

Optional: regenerate the VQC reports and artifacts in `vqc_work/` (about 11 minutes), or run its fast self-test (seconds):

```bash
python vqc_work/scripts/train_vqc.py --selftest
```

```bash
python vqc_work/scripts/train_vqc.py
```

## Files

| Path | What it is |
|---|---|
| `solution.ipynb` | Runnable notebook: PS-Q1 + Task A + Task B, with outputs |
| `processor_d_config.json` | Processor D design; `benchmark_results` equal the official re-run |
| `predictions_q1.csv` | Test predictions (`sample_id,label`), passes `tools/validate_predictions.py` |
| `results.md` | Report following the organizer's `results_template.md` (kit v3) |
| `vqc_work/scripts/train_vqc.py` | Quantum-model code used by the notebook; standalone pipeline and `--selftest` |
| `vqc_work/reports/`, `vqc_work/artifacts/` | Detailed VQC reports, resource summary, saved model state and training history |
| `scripts/train_classical_baseline.py`, `reports/classical_baseline_metrics.csv` | Classical logistic-regression baseline (validation-tuned) and its grid results |
| `data/q1/`, `noise_model/`, `benchmark/`, `tools/`, `schemas/`, `requirements.txt` | Unmodified organizer kit v3 files needed to run (checked against the kit's MANIFEST.sha256) |

## Rules followed

- Only PS-Q1 data is used. Scaler, PCA and the angle scale are fitted on `ecg_train.csv` only; `ecg_val.csv` is used for model selection only; the test features are read once, for inference, after the model is fixed.
- Every noisy run uses `noise_model/benchmark_noise_model.py` unchanged, with `prepare_for_noisy_run()` applied to each executed circuit (basis `rx, ry, rz, cx`).
- Task A and Task B numbers come from `benchmark/qff_qml_benchmark.py`.
- Seed 42 everywhere (plus documented extra seeds 43-46 for robustness checks).

## Environment (`pip freeze`)

Produced with Python 3.13.13 in a venv built from `requirements.txt` (also saved as `pip_freeze.txt`):

```text
annotated-types==0.8.0
anyio==4.15.1
appdirs==1.4.4
appnope==1.0.0
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
asttokens==3.0.2
astunparse==1.6.3
async-lru==2.4.0
attrs==26.1.0
autograd==1.9.1
autoray==0.8.2
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
cachetools==7.2.7
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
comm==0.2.3
contourpy==1.4.0
cryptography==50.0.2
cycler==0.12.1
debugpy==1.8.22
defusedxml==0.7.1
diastatic-malt==2.15.2
dill==0.4.1
executing==2.3.0
fastjsonschema==2.22.2
fonttools==4.68.0
fqdn==1.6.0
gast==0.7.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
ibm-cloud-sdk-core==3.28.1
ibm-platform-services==0.77.1
idna==3.20
ipykernel==7.4.0
ipython==9.17.1
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
jedi==0.20.1
Jinja2==3.1.6
joblib==1.6.0
json5==0.17.3
jsonpointer==3.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_builder==1.2.4
jupyter_client==8.10.0
jupyter_core==5.9.1
jupyter_server==2.21.1
jupyter_server_terminals==0.5.4
jupyterlab==4.6.5
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.1
kiwisolver==1.5.1
lark==1.3.1
MarkupSafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
mistune==3.4.0
mpmath==1.3.0
narwhals==2.28.0
nbclient==0.11.0
nbconvert==7.17.2
nbformat==5.11.1
nest-asyncio2==1.7.4
networkx==3.7
notebook==7.6.3
notebook_shim==0.2.4
numpy==2.4.6
packaging==26.3
pandas==3.1.0
pandocfilters==1.5.1
parso==0.8.7
pennylane==0.44.1
PennyLane-qiskit==0.44.1
pennylane_lightning==0.45.0
pexpect==4.9.0
pillow==13.0.0
platformdirs==4.14.0
prometheus_client==0.26.0
prompt_toolkit==3.0.53
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.4
pycparser==3.12
pydantic==2.14.2
pydantic_core==2.50.2
Pygments==2.21.0
PyJWT==2.15.1
pyparsing==3.3.3
pyspnego==0.12.4
python-dateutil==3.0.0
python-json-logger==4.2.0
PyYAML==6.0.3
pyzmq==27.2.0
qiskit==2.3.0
qiskit-aer==0.17.2
qiskit-ibm-runtime==0.45.1
referencing==0.37.0
requests==2.34.2
requests_ntlm==1.3.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.9.1
rustworkx==0.18.1
scikit-learn==1.9.1
scipy==1.17.1
scipy-openblas32==0.3.34.237.2
Send2Trash==2.1.0
six==1.17.0
soupsieve==3.0.4
stack-data==0.6.3
stevedore==5.9.1
sympy==1.14.0
termcolor==3.3.0
terminado==0.18.1
threadpoolctl==3.7.0
tinycss2==1.5.1
tomlkit==0.15.2
tornado==6.6
traitlets==5.16.1
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.6
uri-template==1.3.0
urllib3==2.8.0
wcwidth==0.9.4
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
wheel==0.48.0
```
