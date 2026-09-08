# Architecture overview

Q-RadAssist is a research prototype for multi-label chest X-ray analysis and
grounded report generation. Phase 0 establishes structure, configuration, and
interfaces only. No models, datasets, QML circuits, or LLM calls are implemented.

## Intended flow

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

The LLM does not diagnose the raw X-ray. It converts structured pipeline
outputs into a human-readable report.

The QML component is experimental. It is not a replacement for the classical
image-analysis pipeline.

## Extensible disease set

Labels live in `config/diseases.json` and are loaded by `DiseaseRegistry`.
Application code should iterate the registry (N diseases) rather than
hard-coding the initial four labels.

Adding a disease to the registry:

- makes the label available to configuration-aware code
- does **not** mean a model can detect that disease
- requires appropriate data, retraining, and evaluation in later phases

The task is **multi-label**: one image may have multiple positive findings.
It is not mutually exclusive multiclass classification.

## Package layout (Phase 0)

| Path | Role |
| --- | --- |
| `config/` | Disease registry and project settings |
| `src/registry.py` | Load and query diseases |
| `src/settings.py` | Load `settings.yaml` |
| `src/contracts.py` | Prediction / findings data contract |
| `src/preprocessing/` | Image validation and preparation (later) |
| `src/models/classical/` | Multi-label classifier (later) |
| `src/models/qml/` | Experimental QML branch (later) |
| `src/xai/` | Explainability (later) |
| `src/reporting/` | Grounded LLM reporting (later) |
| `src/evaluation/` | Metrics and experiment comparison (later) |
| `src/pipeline/` | Orchestration (later) |
| `api/` | Backend HTTP surface (later) |
| `frontend/` | User interface (later) |

## Assumptions (Phase 0)

- Python 3.10+ is the implementation language.
- Configuration is YAML + JSON; no database is required yet.
- Trained weights, datasets, and generated reports are not stored in git
  (see `.gitignore` and directory placeholders).
- Model architecture, QML stack, LLM provider, and datasets are undecided.

See also:

- [Component boundaries](component-boundaries.md)
- [Data contracts](data-contracts.md)
- [Research questions](../research/research-questions.md)
- [Project boundaries](../research/project-boundaries.md)
