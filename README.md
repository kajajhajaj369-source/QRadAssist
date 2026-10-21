# Q-RadAssist

Explainable hybrid AI and quantum machine learning framework for multi-label
chest X-ray analysis and intelligent report generation.

This repository is a research-oriented medical decision-support prototype. It
is not a clinical product.

**Current status: Phase 1A (dataset selection, documentation only).**
Phase 0 foundation is in place. CheXpert is the selected primary dataset.
The dataset has not been downloaded. Preprocessing, training, QML, LLM, API,
and frontend work have not started.

## Short description

Q-RadAssist is designed to analyze a chest X-ray with a classical multi-label
classifier, attach explainability information, optionally compare an
experimental hybrid quantum-classical branch with a classical approach, and
convert structured findings into a grounded natural-language report. The LLM
does not diagnose the raw image. The QML component does not replace the
classical pipeline.

## Current architecture

```
Chest X-ray
    → Image validation & preprocessing
    → Classical deep-learning multi-label classifier
    → Disease predictions
    → XAI explanation
    → Classical feature representation
    → Hybrid quantum-classical / QML branch
    → Comparison of classical vs QML results
    → Structured findings
    → Grounded LLM report generation
    → Web interface
```

Phase 0 established configuration, a disease registry, data contracts, and
component boundaries. Phase 1A records the CheXpert-based core task in
documentation only. No model training, dataset download, QML, LLM, API
server, or frontend is included yet.

Architecture notes: [`docs/architecture/overview.md`](docs/architecture/overview.md).

## Initial disease labels

The system is **multi-label**: one X-ray may have several positive findings.
Labels are not mutually exclusive.

The initial **core training labels** (CheXpert, multi-label) are:

| id | display name |
| --- | --- |
| `pneumonia` | Pneumonia |
| `atelectasis` | Atelectasis |
| `pleural_effusion` | Pleural Effusion |

COVID-19 is **not** in the core training set. It remains in
`config/diseases.json` as an extensible registry entry for a possible later
external/generalization experiment. Adding a disease to the registry does not
train a model or make that disease detectable.

See [`docs/research/dataset-selection.md`](docs/research/dataset-selection.md).

## Major components

| Component | Responsibility |
| --- | --- |
| Preprocessing | Validates and prepares images; does not diagnose |
| Classical model | Multi-label classification scores; does not write reports |
| XAI | Explains model behavior; does not establish clinical truth |
| QML | Experimental hybrid branch for later comparison |
| Structured findings | Intermediate representation; does not make new predictions |
| LLM reporting | Grounded language from findings; does not diagnose raw X-rays |
| Backend | Orchestration, separate from model internals |
| Frontend | Presentation only; no ML logic |

Details: [`docs/architecture/component-boundaries.md`](docs/architecture/component-boundaries.md).

## Research questions

1. How effectively can a deep-learning model perform multi-label classification of selected chest X-ray abnormalities?
2. Can XAI techniques provide interpretable visual evidence for the model's predictions?
3. How does a hybrid quantum-classical classifier compare with a classical classifier for the selected chest X-ray classification task?
4. Can structured model predictions and explainability information be converted into a useful natural-language report while minimizing unsupported content?

See [`docs/research/research-questions.md`](docs/research/research-questions.md).

## Development phases

| Phase | Scope |
| --- | --- |
| **0** | Repository foundation, registry, config, contracts, docs, basic tests (complete) |
| **1A** | Primary dataset selection and core-label documentation (current) |
| 1B | Dataset access, preprocessing, and split design (not started) |
| 2 | Classical multi-label model (not started) |
| 3 | XAI (not started) |
| 4 | Experimental QML comparison (not started) |
| 5 | Grounded reporting (not started) |
| 6 | Interface and integration (not started) |

Later phase internals remain unspecified except where Phase 1A documents the
dataset decision.

## Data

**Primary dataset (selected, not downloaded): CheXpert.**

The project uses **real-world medical imaging datasets**. AI-generated X-rays
are **not** part of the core dataset.

Rationale, COVID-19 exclusion, and known dataset risks:
[`docs/research/dataset-selection.md`](docs/research/dataset-selection.md).

## Running tests

Python 3.10+ is assumed.

```bash
python -m pip install -r requirements.txt
python -m pytest
```

Phase 0 dependencies are PyYAML (settings) and pytest (tests) only.

## Medical / research disclaimer

This project will not claim to replace radiologists, provide definitive
clinical diagnoses, or guarantee clinical accuracy. It will not claim quantum
advantage without experimental evidence. An XAI heatmap does not prove
disease. The LLM does not independently diagnose an X-ray.

See [`docs/research/project-boundaries.md`](docs/research/project-boundaries.md).

## Installation & Quick Start

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/kajajhajaj369-source/QRadAssist.git
cd QRadAssist

# Create a clean virtual environment (recommended)
python -m venv .venv_cuda
.venv_cuda\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the API locally

```bash
# Start the FastAPI server (default port 8003)
uvicorn src.api.app:app --host 127.0.0.1 --port 8003
```

The server provides the following endpoints:
- `GET /health` – health check.
- `GET /labels` – list of disease labels expected by the model.
- `POST /predict` – single‑image inference, returns sigmoid probabilities.
- `POST /analyze` – full analysis (probabilities, draft report, Grad‑CAM heatmaps for **all** target diseases).

## Model checkpoints

Two checkpoint files are shipped with the repository and are **required** for inference:
- `src/models/classical/best_model.pt` – classical DenseNet‑121 multi‑label model.
- `experiments/full_gpu/best_full_model.pt` – full‑GPU trained version (identical architecture, different training run).

Both files are under 30 MiB and are stored directly in the repository (no Git LFS needed). They are **read‑only** and should not be modified.

## Running the test suite

```bash
pytest -q
```
All unit tests should pass on a clean environment.

## Disclaimer (re‑stated)

This project is a **research prototype** and **not** a clinically validated diagnostic tool. Predictions, Grad‑CAM visualisations, and generated reports are **assistive** only and must be reviewed by qualified medical professionals before any clinical use.
