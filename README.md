# QRadAssist

Explainable hybrid AI and quantum machine learning framework for multi-label
chest X-ray analysis and intelligent report generation.

This repository is a research-oriented medical decision-support prototype.
It is not a clinical product.

**Current status: Phase 0 (project foundation).** Later phases have not started.

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
    -> Image validation & preprocessing
    -> Classical deep-learning multi-label classifier
    -> Disease predictions
    -> XAI explanation
    -> Classical feature representation
    -> Hybrid quantum-classical / QML branch
    -> Comparison of classical vs QML results
    -> Structured findings
    -> Grounded LLM report generation
    -> Web interface
```

Phase 0 implements configuration, a disease registry, data contracts, and
documentation of these boundaries. No model training, dataset download, QML,
LLM, API server, or frontend is included yet.

Architecture notes: [docs/architecture/overview.md](docs/architecture/overview.md).

## Initial disease labels

The system is **multi-label**: one X-ray may have several positive findings.
Labels are not mutually exclusive.

Initial registry (`config/diseases.json`):

| id | display name |
| --- | --- |
| `pneumonia` | Pneumonia |
| `covid19` | COVID-19 |
| `pleural_effusion` | Pleural Effusion |
| `atelectasis` | Atelectasis |

The disease architecture is extensible. Application code reads N diseases from
the registry instead of hard-coding this list. Adding a disease to the
registry does not train a model or make that disease detectable.

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

Details: [docs/architecture/component-boundaries.md](docs/architecture/component-boundaries.md).

## Research questions

1. How effectively can a deep-learning model perform multi-label classification of selected chest X-ray abnormalities?
2. Can XAI techniques provide interpretable visual evidence for the model's predictions?
3. How does a hybrid quantum-classical classifier compare with a classical classifier for the selected chest X-ray classification task?
4. Can structured model predictions and explainability information be converted into a useful natural-language report while minimizing unsupported content?

See [docs/research/research-questions.md](docs/research/research-questions.md).

## Development phases

| Phase | Scope |
| --- | --- |
| **0** | Repository foundation, registry, config, contracts, docs, basic tests |
| 1 | Dataset strategy, preprocessing design, evaluation plan (not started) |
| 2 | Classical multi-label model (not started) |
| 3 | XAI (not started) |
| 4 | Experimental QML comparison (not started) |
| 5 | Grounded reporting (not started) |
| 6 | Interface and integration (not started) |

Phases 1-6 are planning labels only. Their internals are not specified here.

## Data

Later work will use **real-world medical imaging datasets**. AI-generated
X-rays are **not** part of the core dataset. Datasets are not downloaded in
Phase 0.

## Running Phase 0 tests

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

See [docs/research/project-boundaries.md](docs/research/project-boundaries.md).
