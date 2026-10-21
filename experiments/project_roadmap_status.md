# QRadAssist — Original 11-Phase Project Roadmap & Status Report

**Report Generated:** 2026-10-21  
**Target Repository:** `E:\College\QRadAssist`  
**Audited Directory:** `experiments/`  
**Report Type:** Phase Roadmap Recovery and Project Status Audit  

---

> [!IMPORTANT]
> **Roadmap Provenance & Audit Methodology:**  
> A comprehensive scan was performed across all documentation (`docs/`), planning files, `README.md`, source tree comments, configuration files (`config/`), and the historical task/conversation transcript (steps 0 through 2498).
> No monolithic file in the repository was titled "11-phase roadmap". Instead, the 11 phases represent the project's sequence reconstructed from:
> 1. The primary architectural development table in [`README.md`](file:///E:/College/QRadAssist/README.md) (listing Phases 0 through 6).
> 2. The explicit sub-phase definitions and research protocols in [`docs/research/`](file:///E:/College/QRadAssist/docs/research/) (Phases 1A, 1B, 1C, 1D, 2A, 2B, 2C).
> 3. The user's explicit multi-phase operational directives: "Phase 9" for full-dataset GPU training (step 1879), "Phase 2" of final evaluation (step 2085), and "Phase 3" of explainability & demo validation (step 2321).
> 
> Below, these milestones are mapped to an 11-phase sequence with full evidence, artifacts, and test verification.

---

## 1. Recovered 11 Phases: Overview & Status

| Phase # | Phase Identifier | Phase Name & Scope | Status | Primary Evidence & Artifacts |
|:---:|:---:|:---|:---:|:---|
| **1** | **Phase 0** | Repository Foundation, Config, Contracts & Core Architecture | **Complete** | `src/contracts.py`, `src/registry.py`, `src/settings.py`, `config/` |
| **2** | **Phase 1A** | Primary Dataset Selection & Feasibility Analysis | **Complete** | `docs/research/dataset-selection.md` |
| **3** | **Phase 1B** | Dataset Acquisition, Hierarchy & Exploration | **Complete** | `docs/research/dataset-inspection.md`, `data/raw/chexpert-small/` |
| **4** | **Phase 1C** | Dataset Integrity, Path Resolution & Leakage Verification | **Complete** | `src/evaluation/dataset_integrity.py`, `docs/research/dataset-integrity.md` |
| **5** | **Phase 1D** | Experimental Protocol Design & Uncertainty Policy Definition | **Complete** | `src/evaluation/dataset_protocol_analysis.py`, `docs/research/dataset-protocol.md` |
| **6** | **Phase 2A** | Deterministic Preprocessing Pipeline & Transforms | **Complete** | `src/preprocessing/transforms.py`, `src/preprocessing/labels.py` |
| **7** | **Phase 2B** | Manifest Splitting & Patient-Isolated Cohort Materialization | **Complete** | `src/preprocessing/materialize.py`, `data/splits/*.csv` |
| **8** | **Phase 2C** | Exact Normalization Statistics Computation | **Complete** | `src/preprocessing/compute_stats.py`, `data/splits/training_statistics.json` |
| **9** | **Phase 9** | Full-Dataset GPU Multi-Label Training & Checkpointing | **Complete** | `train_full.py`, `experiments/full_gpu/best_full_model.pt` |
| **10** | **Phase 10** | Final Held-Out Evaluation, Metric Recomputation & Split Audit | **Complete** | `experiments/final_evaluation/evaluation_report.md`, `evaluation_audit.md` |
| **11** | **Phase 11** | Explainability (Grad-CAM), Demo Validation & Reporting Pipeline | **Complete** | `src/models/classical/xai.py`, `src/api/app.py`, `src/frontend/` |

---

## 2. Detailed Phase Evidence, Deliverables, and Audit

### Phase 1: Phase 0 — Repository Foundation & Core Contracts
- **Status:** `Complete`
- **Objectives:** Establish project directory hierarchy, type-safe data contracts, disease registries, configuration management, and baseline test suite.
- **Evidence & File Paths:**
  - Data contracts: [`src/contracts.py`](file:///E:/College/QRadAssist/src/contracts.py) (`DiseasePrediction`, `XAIAttachment`, `AnalysisFindings`).
  - Registry: [`src/registry.py`](file:///E:/College/QRadAssist/src/registry.py) loading [`config/diseases.json`](file:///E:/College/QRadAssist/config/diseases.json).
  - Settings: [`src/settings.py`](file:///E:/College/QRadAssist/src/settings.py) loading [`config/settings.yaml`](file:///E:/College/QRadAssist/config/settings.yaml).
  - Tests: [`tests/test_contracts.py`](file:///E:/College/QRadAssist/tests/test_contracts.py), [`tests/test_disease_registry.py`](file:///E:/College/QRadAssist/tests/test_disease_registry.py), [`tests/test_settings.py`](file:///E:/College/QRadAssist/tests/test_settings.py).
- **Completed Deliverables:** Type-safe model output containers, disease configuration, path settings, initial unit tests.
- **Remaining Tasks:** None.

---

### Phase 2: Phase 1A — Primary Dataset Selection & Feasibility
- **Status:** `Complete`
- **Objectives:** Evaluate candidate chest X-ray datasets (CheXpert, MIMIC-CXR, NIH ChestX-ray14) and justify selection against research requirements.
- **Evidence & File Paths:**
  - Documentation: [`docs/research/dataset-selection.md`](file:///E:/College/QRadAssist/docs/research/dataset-selection.md).
- **Completed Deliverables:** Formal decision selecting Stanford CheXpert (chexpert-small) based on realistic uncertainty labels and standardized frontal/lateral views. COVID-19 excluded from primary training due to zero CheXpert coverage.
- **Remaining Tasks:** None.

---

### Phase 3: Phase 1B — Dataset Acquisition & Exploration
- **Status:** `Complete`
- **Objectives:** Download and inspect the CheXpert dataset; catalog image resolutions, views, patient distributions, and raw label frequencies.
- **Evidence & File Paths:**
  - Raw data: `data/raw/chexpert-small/` (223,414 training rows, 234 validation rows).
  - Documentation: [`docs/research/dataset-inspection.md`](file:///E:/College/QRadAssist/docs/research/dataset-inspection.md).
- **Completed Deliverables:** Characterization of 191,027 frontal and 32,387 lateral images; study encounter structure; class imbalance documentation.
- **Remaining Tasks:** None.

---

### Phase 4: Phase 1C — Dataset Integrity & Leakage Verification
- **Status:** `Complete`
- **Objectives:** Verify image file existence on disk, detect missing or corrupted images, and prove zero patient leakage across initial partitions.
- **Evidence & File Paths:**
  - Verification tool: [`src/evaluation/dataset_integrity.py`](file:///E:/College/QRadAssist/src/evaluation/dataset_integrity.py).
  - Documentation: [`docs/research/dataset-integrity.md`](file:///E:/College/QRadAssist/docs/research/dataset-integrity.md).
  - Tests: [`tests/test_dataset_integrity.py`](file:///E:/College/QRadAssist/tests/test_dataset_integrity.py).
- **Completed Deliverables:** 100% file presence verified (all 223,414 images resolvable); zero patient overlap confirmed between official train and validation sets.
- **Remaining Tasks:** None.

---

### Phase 5: Phase 1D — Experimental Dataset Protocol & Uncertainty Policy
- **Status:** `Complete`
- **Objectives:** Define the formal experimental protocol, target disease subset, view constraints, and handling of uncertain (`-1.0`) labels.
- **Evidence & File Paths:**
  - Protocol analysis script: [`src/evaluation/dataset_protocol_analysis.py`](file:///E:/College/QRadAssist/src/evaluation/dataset_protocol_analysis.py).
  - Protocol documentation: [`docs/research/dataset-protocol.md`](file:///E:/College/QRadAssist/docs/research/dataset-protocol.md).
  - Tests: [`tests/test_dataset_protocol_analysis.py`](file:///E:/College/QRadAssist/tests/test_dataset_protocol_analysis.py).
- **Completed Deliverables:**
  - Selection of 3 target conditions: Atelectasis, Cardiomegaly/Pneumonia/Pleural Effusion (primary clinical targets: Atelectasis, Pleural Effusion, Pneumonia).
  - Selection of frontal-only views (PA and AP).
  - Uncertainty policy: `U-Ignore` baseline masking out `-1.0` labels in BCEWithLogitsLoss.
  - Policy reserving official validation set strictly for final evaluation benchmark.
- **Remaining Tasks:** None.

---

### Phase 6: Phase 2A — Preprocessing Pipeline Implementation
- **Status:** `Complete`
- **Objectives:** Implement deterministic preprocessing: PIL grayscale verification, resizing to 224x224, min-max scaling, label parsing, and isolated training augmentations.
- **Evidence & File Paths:**
  - Implementation: [`src/preprocessing/transforms.py`](file:///E:/College/QRadAssist/src/preprocessing/transforms.py), [`src/preprocessing/labels.py`](file:///E:/College/QRadAssist/src/preprocessing/labels.py), [`src/preprocessing/manifest.py`](file:///E:/College/QRadAssist/src/preprocessing/manifest.py).
  - Documentation: [`docs/research/preprocessing-pipeline.md`](file:///E:/College/QRadAssist/docs/research/preprocessing-pipeline.md).
  - Tests: [`tests/test_preprocessing.py`](file:///E:/College/QRadAssist/tests/test_preprocessing.py) (32 passing tests).
- **Completed Deliverables:** `DeterministicPreprocessor`, `TrainingAugmentor`, `CheXpertLabelParser`, manifest serializing/deserializing routines.
- **Remaining Tasks:** None.

---

### Phase 7: Phase 2B — Dataset Splits Materialization
- **Status:** `Complete`
- **Objectives:** Partition the frontal cohort at the patient level to eliminate data leakage; materialize deterministic manifest CSV files.
- **Evidence & File Paths:**
  - Splitting logic: [`src/preprocessing/splitting.py`](file:///E:/College/QRadAssist/src/preprocessing/splitting.py), [`src/preprocessing/materialize.py`](file:///E:/College/QRadAssist/src/preprocessing/materialize.py).
  - Documentation: [`docs/research/dataset-split-and-cohort.md`](file:///E:/College/QRadAssist/docs/research/dataset-split-and-cohort.md).
  - Materialized manifests:
    - `data/splits/train_manifest.csv` (172,033 images, 58,081 patients)
    - `data/splits/val_manifest.csv` (18,994 images, 6,453 patients)
    - `data/splits/dev_cohort_manifest.csv` (15,000 images, 5,084 patients)
    - `data/splits/train_val_split.csv` (191,027 frontal images)
  - Tests: [`tests/test_materialized_splits.py`](file:///E:/College/QRadAssist/tests/test_materialized_splits.py) (10 passing tests).
- **Completed Deliverables:** Deterministic patient-split manifests with zero overlap verified by automated tests.
- **Remaining Tasks:** None.

---

### Phase 8: Phase 2C — Normalization Statistics Computation
- **Status:** `Complete`
- **Objectives:** Compute exact pixel-level mean and standard deviation across all pixels in the training manifest using streaming Welford/moment accumulator.
- **Evidence & File Paths:**
  - Computation script: [`src/preprocessing/compute_stats.py`](file:///E:/College/QRadAssist/src/preprocessing/compute_stats.py).
  - Output artifact: `data/splits/training_statistics.json` ($\mu = 0.506422$, $\sigma = 0.286777$ across 8,631,927,808 pixels).
  - Documentation: [`docs/research/preprocessing-statistics.md`](file:///E:/College/QRadAssist/docs/research/preprocessing-statistics.md).
  - Configuration lock: [`config/settings.yaml`](file:///E:/College/QRadAssist/config/settings.yaml).
- **Completed Deliverables:** Exact dataset statistics locked into configuration.
- **Remaining Tasks:** None.

---

### Phase 9: Phase 9 — Full-Dataset GPU Training
- **Status:** `Complete`
- **Objectives:** Execute multi-epoch GPU training on NVIDIA hardware using DenseNet-121, BCEWithLogitsLoss with U-Ignore masking, atomic checkpointing, and persistent training logging.
- **Evidence & File Paths:**
  - Training driver: [`train_full.py`](file:///E:/College/QRadAssist/train_full.py).
  - Classical modules: [`src/models/classical/model.py`](file:///E:/College/QRadAssist/src/models/classical/model.py), [`src/models/classical/training.py`](file:///E:/College/QRadAssist/src/models/classical/training.py), [`src/models/classical/dataset.py`](file:///E:/College/QRadAssist/src/models/classical/dataset.py), [`src/models/classical/loss.py`](file:///E:/College/QRadAssist/src/models/classical/loss.py).
  - Training logs: `experiments/full_gpu/training.log`.
  - Trained model: `experiments/full_gpu/best_full_model.pt` (SHA-256: `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a`).
  - Epoch checkpoints: `experiments/full_gpu/epoch_1_ckpt.pt` through `epoch_5_ckpt.pt`.
  - Production checkpoint isolation: `src/models/classical/best_model.pt` (SHA-256: `392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a`) remained 100% untouched and unpromoted.
  - Tests: [`tests/test_training_recovery.py`](file:///E:/College/QRadAssist/tests/test_training_recovery.py) (15 passed, 1 skipped).
- **Validation Results (Best Epoch 3):**
  - Pneumonia AUROC: 0.7869
  - Atelectasis AUROC: 0.7136
  - Pleural Effusion AUROC: 0.8898
  - Mean Validation AUROC: **0.7968**
- **Completed Deliverables:** Fully trained multi-label CNN, checkpoint recovery logic, atomic save guarantees.
- **Remaining Tasks:** None.

---

### Phase 10: Phase 10 — Final Held-Out Evaluation & Integrity Audit
- **Status:** `Complete`
- **Objectives:** Evaluate `experiments/full_gpu/best_full_model.pt` on the held-out test cohort (202 frontal images, 200 patients), independently recompute metrics, calculate patient-clustered bootstrap confidence intervals, and audit patient separation.
- **Evidence & File Paths:**
  - Evaluation reports: [`experiments/final_evaluation/evaluation_report.md`](file:///E:/College/QRadAssist/experiments/final_evaluation/evaluation_report.md), [`experiments/final_evaluation/evaluation_audit.md`](file:///E:/College/QRadAssist/experiments/final_evaluation/evaluation_audit.md).
  - Metric outputs: `experiments/final_evaluation/test_metrics.json`, `experiments/final_evaluation/test_metrics.csv`.
- **Test Metric Deliverables (N=202 frontal images, 200 patients):**
  - Pneumonia: AUROC **0.8183** (95% CI: [0.7109, 0.9188]), AP 0.1588
  - Atelectasis: AUROC **0.8076** (95% CI: [0.7433, 0.8662]), AP 0.6960
  - Pleural Effusion: AUROC **0.9358** (95% CI: [0.9020, 0.9646]), AP 0.8754
  - Macro AUROC: **0.8539**, Macro AP: **0.5767**
- **Audit Findings:**
  - Zero patient overlap confirmed between train/val manifests and test split via exhaustive set intersection.
  - Discrepancy between 202 images and 200 patients resolved (patients 64547 and 64581 had 2 repeat PA views in study1).
  - Critical clinical notice: Pneumonia sensitivity is 0.0 at default 0.5 threshold due to uncalibrated logits and low prevalence (operating threshold must be calibrated before clinical use).
- **Remaining Tasks:** None for the audit phase; threshold calibration recommended before clinical deployment.

---

### Phase 11: Phase 11 — Explainability, Demo Validation & API Integration
- **Status:** `Complete`
- **Objectives:** Validate Grad-CAM heatmaps on DenseNet-121 layer4 features, fix PyTorch hook deprecation, integrate `QRADASSIST_CHECKPOINT_PATH` override, enforce single-sigmoid application, add clinical draft report disclaimer, and conduct end-to-end API testing.
- **Evidence & File Paths:**
  - Explainability: [`src/models/classical/xai.py`](file:///E:/College/QRadAssist/src/models/classical/xai.py) (Grad-CAM with tensor output hook).
  - FastAPI server: [`src/api/app.py`](file:///E:/College/QRadAssist/src/api/app.py) (endpoints `/health`, `/predict`, `/report`, `/gradcam`, `/labels`).
  - Frontend: [`src/frontend/index.html`](file:///E:/College/QRadAssist/src/frontend/index.html), [`src/frontend/script.js`](file:///E:/College/QRadAssist/src/frontend/script.js), [`src/frontend/style.css`](file:///E:/College/QRadAssist/src/frontend/style.css).
  - Validation report: [`experiments/phase3/phase3_validation_report.md`](file:///E:/College/QRadAssist/experiments/phase3/phase3_validation_report.md).
  - Test suites: [`tests/test_xai.py`](file:///E:/College/QRadAssist/tests/test_xai.py), [`tests/test_api.py`](file:///E:/College/QRadAssist/tests/test_api.py), [`tests/test_report.py`](file:///E:/College/QRadAssist/tests/test_report.py).
- **Completed Deliverables:**
  - Modernized Grad-CAM implementation producing distinct class-specific heatmaps (MAD > 0.22).
  - Optional `QRADASSIST_CHECKPOINT_PATH` env var allowing evaluation of experimental models without touching production checkpoint.
  - Prominent UI disclaimer: "⚠️ AI-GENERATED DRAFT REPORT — REQUIRES QUALIFIED CLINICIAN REVIEW".
  - Sigmoid applied strictly once; `/gradcam` receives raw logits.
  - Full test suite passing (38 passed, 1 skipped).
- **Remaining Tasks:** None for this phase.

---

## 3. High-Level Component Roadmap vs. Executed Phases

In the project's original [`README.md`](file:///E:/College/QRadAssist/README.md), the architectural lifecycle was structured by major sub-systems:
- **Phase 0:** Foundation, Config, and Contracts (✅ Completed in Phase 1/Phase 0)
- **Phase 1:** Dataset Protocol & Audit (✅ Completed in Phases 1A–1D)
- **Phase 2:** Preprocessing & Baseline Splitting (✅ Completed in Phases 2A–2C)
- **Phase 3:** Classical Baseline Multi-Label Model (✅ Completed in Phase 9, evaluated in Phase 10)
- **Phase 4:** Hybrid Quantum Machine Learning (QML) Branch (⚠️ Partially Implemented / Stubs only)
- **Phase 5:** Grounded LLM Report Generation (⚠️ Rule-based only / LLM stub only)
- **Phase 6:** End-to-End Pipeline, API & Dashboard (✅ Completed in Phase 11)

### Status of Remaining High-Level Research Tracks

1. **Hybrid Quantum Machine Learning (QML) Branch:**
   - **Current State:** Stubs and experimental layers exist in [`src/models/classical/qml.py`](file:///E:/College/QRadAssist/src/models/classical/qml.py) (`HybridQuantumClassifier`, `SimpleQuantumLayer`). Package [`src/models/qml/__init__.py`](file:///E:/College/QRadAssist/src/models/qml/__init__.py) and [`tests/test_qml.py`](file:///E:/College/QRadAssist/tests/test_qml.py) exist.
   - **Gap:** No training run has been executed on the QML branch; no PennyLane VQC circuit has been trained on bottleneck embeddings; no empirical comparison against classical DenseNet-121 has been recorded.
   - **Status:** `Pending` / `In Progress`

2. **Grounded LLM Report Generation:**
   - **Current State:** [`src/models/classical/report.py`](file:///E:/College/QRadAssist/src/models/classical/report.py) implements rule-based deterministic template reports from `AnalysisFindings`. [`src/reporting/__init__.py`](file:///E:/College/QRadAssist/src/reporting/__init__.py) is an empty stub.
   - **Gap:** No external or local LLM prompting (e.g. via instruction-tuned open weights or API) has been connected to generate synthesized impressions.
   - **Status:** `Pending`

3. **Operating Threshold Calibration:**
   - **Current State:** Evaluated at arbitrary 0.5 threshold, resulting in 0% Pneumonia sensitivity despite high AUROC (0.8183).
   - **Gap:** Validation-based threshold tuning (e.g. seeking 80–90% sensitivity on validation split) has not yet been locked into `config/diseases.json`.
   - **Status:** `Pending`

---

## 4. Current Test Suite Status

A total of **93 tests** are currently defined across 14 test modules:
```
tests/test_api.py (10 tests)
tests/test_contracts.py (3 tests)
tests/test_dataset_integrity.py (4 tests)
tests/test_dataset_protocol_analysis.py (4 tests)
tests/test_disease_registry.py (3 tests)
tests/test_materialized_splits.py (10 tests)
tests/test_phase2_modules.py (10 tests)
tests/test_preprocessing.py (18 tests)
tests/test_qml.py (2 tests)
tests/test_report.py (4 tests)
tests/test_settings.py (1 test)
tests/test_training_recovery.py (17 tests)
tests/test_xai.py (7 tests)
```
Last regression execution on Phase 11 completion: **38 passed, 1 skipped, 0 failed (exit code 0)**.

---

## 5. Recommended Next Phase

Based on the original project dependencies and academic/EDI goals, the recommended next sequence is:

1. **Next Immediate Step: QML Test & Baseline Verification (README Phase 4)**
   - Run and audit [`tests/test_qml.py`](file:///E:/College/QRadAssist/tests/test_qml.py).
   - Train the `HybridQuantumClassifier` on the 15,000-image development cohort (`data/splits/dev_cohort_manifest.csv`).
   - Evaluate against classical DenseNet-121 to answer the primary research question regarding hybrid quantum utility.

2. **Secondary Step: Clinical Decision Threshold Calibration**
   - Calibrate pathology-specific decision thresholds on `data/splits/val_manifest.csv` to ensure clinical sensitivity (especially for Pneumonia) before final project demonstration.

3. **Tertiary Step: Grounded LLM Report Generation (README Phase 5)**
   - Connect `src/reporting/` to an instruction-tuned model for drafting fluent findings from `AnalysisFindings` while preventing diagnostic hallucination.

---

## 6. Audit & Safety Invariants Verified

- **Protected Production Checkpoint:** `src/models/classical/best_model.pt`  
  - SHA-256: `392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a` (Untouched, unpromoted).
- **Experimental Trained Checkpoint:** `experiments/full_gpu/best_full_model.pt`  
  - SHA-256: `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a` (Intact).
- **No dataset files, manifests, or checkpoints were modified or overwritten during this inspection.**
