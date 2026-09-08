# Component boundaries

These boundaries are the intended responsibilities of each major component.
Phase 0 documents them; implementations come later.

## Image preprocessing

- Validates and prepares chest X-ray images.
- Does not diagnose.

## Classical model

- Performs multi-label image classification.
- Produces prediction scores for registry diseases.
- Does not generate reports.

## XAI

- Explains model behavior (for example, visual evidence for a prediction).
- Does not establish clinical truth.
- A heatmap is not proof of disease.

## QML

- Experimental hybrid quantum-classical component.
- Will later be compared against a classical approach.
- Does not automatically imply quantum advantage.

## Structured findings

- Normalizes outputs from the analysis pipeline.
- Acts as an intermediate representation for reporting and evaluation.
- Does not make new predictions.

## LLM reporting

- Converts structured findings into readable language.
- Must be grounded in supplied findings.
- Must not independently diagnose raw X-rays.
- Must not invent unsupported medical findings.

## Backend

- Eventually orchestrates components (`src/pipeline/` and later `api/`).
- Should remain separate from model implementation.

## Frontend

- Eventually displays results.
- Should not contain ML logic.
