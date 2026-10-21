# QRadAssist — Final Demo-Readiness Audit and EDI Submission Preparation Report

**Project Title:** Q-RadAssist (Explainable Hybrid AI & Quantum Machine Learning Framework for Multi-Label Chest X-Ray Analysis and Intelligent Report Generation)  
**Project Path:** `E:\College\QRadAssist`  
**Audit Date:** 2026-10-21  
**Audit Status:** Complete — Verified for Academic & Engineering Demonstration  

---

## 1. Executive Summary & Overall Readiness Status

The QRadAssist project was subjected to a comprehensive final audit to verify demo-readiness for the Engineering Design and Innovation (EDI) college evaluation.

### Overall Status: **DEMO-READY (Verified)**

| Component | Status | Verification Summary |
|:---|:---:|:---|
| **Model Weights & Checkpoints** | **Verified** | Both checkpoints preserved; SHA-256 hashes strictly unchanged before and after audit. |
| **Unified Single-Upload API** | **Verified** | `POST /analyze` processes radiograph once, returning probabilities, report, and 3 heatmaps in 1 payload. |
| **Interactive Web Frontend** | **Verified** | Dark medical dashboard loaded and visually validated in browser; single-action analysis functioning. |
| **Multi-Class Explainability (Grad-CAM)** | **Verified** | Distinct disease color mapping (Pneumonia: Red/Orange, Atelectasis: Blue/Cyan, Effusion: Green/Lime). |
| **Scientific Disclaimers & Ethics** | **Verified** | Mandatory clinical decision support disclaimers displayed prominently across UI and API payloads. |
| **Automated Test Suite** | **Verified** | 103 passed, 1 skipped, 0 failed across 15 test files (31.28s runtime). |
| **EDI Submission Documentation** | **Requires User Input** | Technical artifacts complete; team details, exact literature citations, and institutional cost data require student input. |

---

## 2. UI/UX Issues Found & Targeted Fixes

Visual and functional inspection of the web application was performed using the live browser session at `http://127.0.0.1:8000/`.

### Verified UI Behaviors:
- **Single-Upload Workflow:** User selects an image once. Preview appears with filename, file size, and "Change Image" button.
- **One-Click Consolidated Results:** Clicking *"Analyze X-Ray & Generate Complete Report"* executes backend inference and reveals the complete analysis dashboard without page reloads.
- **Visual Presentation:** Clean dark-mode medical UI with high visual contrast, semantic typography, and responsive cards.
- **Three Disease-Specific Grad-CAM Overlays:** Visual activations rendered on separate cards with distinctive color overlays:
  - **Pneumonia:** Red / Orange (`#FF4500` / `rgba(255, 69, 0, alpha)`)
  - **Atelectasis:** Blue / Cyan (`#00BFFF` / `rgba(0, 191, 255, alpha)`)
  - **Pleural Effusion:** Green / Lime (`#32CD32` / `rgba(50, 205, 50, alpha)`)
- **Color Legend:** Dedicated mapping bar clearly explains each color channel.

### Specific Deficiencies Identified & Fixed:

| Item | Problem Found | Fix Applied | Status |
|:---|:---|:---|:---:|
| **1. Unsubstantiated Calibration Claim in UI** | `src/frontend/script.js` displayed `<span class="prob-subtext">Calibrated Prob: X.XXXX</span>`. Since model outputs are raw sigmoid values and post-hoc calibration was not performed, this claim violated reporting rules. | Updated `src/frontend/script.js` (line 207) to label the value as `Predicted Prob: ${prob.toFixed(4)}`. Also clarified docstrings in `src/api/app.py`. | **Fixed and retested** |
| **2. Radiograph Aspect Ratio Clipping in Preview** | `.image-preview` in `src/frontend/style.css` used `object-fit: cover`, which clipped non-square radiographs. | Changed to `object-fit: contain; background: #000;` to ensure full image visibility without stretching or clipping. | **Fixed and retested** |
| **3. Grad-CAM Canvas Aspect Ratio Distortion Risk** | `.canvas-wrapper` forced `aspect-ratio: 1 / 1;` with `object-fit: cover` on `.gradcam-base-img`, causing rectangular radiographs to be cropped while canvas overlay could become misaligned. | Changed `.canvas-wrapper` to `min-height: 220px;` and `.gradcam-base-img` to `width: 100%; height: auto; display: block;`. The canvas overlay matches intrinsic image geometry with zero clipping. | **Fixed and retested** |
| **4. Mobile Viewport Wrapping (< 640px)** | On narrow viewports, action buttons and container padding caused minor horizontal scrolling. | Added a mobile media query `@media (max-width: 640px)` in `src/frontend/style.css` ensuring full-width buttons and clean vertical stacking. | **Fixed and retested** |

---

## 3. API & End-to-End Smoke-Test Results

The FastAPI application was audited through direct live requests on `http://127.0.0.1:8000`:

| Endpoint | Method | Test Input | HTTP Status | Response Contract Verification | Classification |
|:---|:---:|:---|:---:|:---|:---:|
| `/` | `GET` | N/A | `200 OK` | Serves HTML frontend (`index.html`), loads CSS (`style.css`) and JS (`script.js`). | **Verified** |
| `/health` | `GET` | N/A | `200 OK` | Returns `{"status": "ok"}`. | **Verified** |
| `/labels` | `GET` | N/A | `200 OK` | Returns exact list `["Pneumonia", "Atelectasis", "Pleural Effusion"]`. | **Verified** |
| `/analyze` | `POST` | Valid frontal radiograph (`patient00027/study1/view1_frontal.jpg`) | `200 OK` | Returns `probabilities`, `raw_logits`, `report`, `heatmaps` (all 3 diseases, 224×224 grids), `color_scheme`, and `disclaimer`. | **Verified** |
| `/analyze` | `POST` | Text file (`dummy.txt`) | `422 Unprocessable Content` | Returns `{"detail": "Uploaded file must be an image."}`. | **Verified** |
| `/analyze` | `POST` | Corrupt image bytes (`bad.jpg`) | `422 Unprocessable Content` | Returns `{"detail": "Uploaded file is not a valid image."}`. | **Verified** |
| `/predict` | `POST` | Valid radiograph | `200 OK` | Legacy endpoint remains backwards-compatible; returns `{"probabilities": {...}}`. | **Verified** |
| `/report` | `POST` | Valid radiograph | `200 OK` | Legacy endpoint returns `{"findings": [...], "impression": "..."}`. | **Verified** |
| `/gradcam` | `POST` | Valid radiograph + `disease="Atelectasis"` | `200 OK` | Legacy endpoint returns `{"heatmap": [...]}`. | **Verified** |

### Single-Sigmoid Guarantee Verification:
- Model raw output: `logits = model(tensor)`
- Probability computation: `probs = torch.sigmoid(logits)` (applied exactly once in backend)
- Draft report generation: `generate_report(probs)` evaluates `prob >= threshold` directly
- Frontend consumption: `normalizeProbability(val)` checks range `[0, 1]` and passes values through without secondary sigmoid activation.

---

## 4. Checkpoint Selection & Cryptographic Verification

### Checkpoint Loading Logic in `src/api/app.py`:
- `_get_checkpoint_path()` inspects the environment variable `QRADASSIST_CHECKPOINT_PATH`.
- **Default (Unset):** Points to `src/models/classical/best_model.pt` (`CHECKPOINT_PATH` constant).
- **Environment Override:** If `QRADASSIST_CHECKPOINT_PATH` is set, uses the specified path.

### Checkpoint Selected for Demo:
- Selected: `experiments/full_gpu/best_full_model.pt` (Full GPU-trained model, 172,033 images, 5 epochs).
- Launched via: `$env:QRADASSIST_CHECKPOINT_PATH = "experiments/full_gpu/best_full_model.pt"`

### Cryptographic Hash Integrity Audit (SHA-256):

| Checkpoint Path | Hash Before Audit | Hash After Audit | Match Status |
|:---|:---|:---|:---:|
| `experiments/full_gpu/best_full_model.pt` | `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a` | `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a` | **Exact Match (Unchanged)** |
| `src/models/classical/best_model.pt` | `392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a` | `392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a` | **Exact Match (Unchanged)** |

*Neither checkpoint was retrained, overwritten, or modified.*

---

## 5. Scientific Findings & Verification Matrix

All metrics and scientific claims presented in the demo and documentation were cross-checked against experimental logs and raw data:

| Metric / Finding | Documented Value | Verified Value | Verification Source | Status |
|:---|:---:|:---:|:---|:---:|
| **Primary Held-Out Cohort** | 202 images / 200 patients | 202 images / 200 patients | `experiments/final_evaluation/evaluation_report.md` | **Verified** |
| **Primary Macro AUROC** | 0.8539 | 0.8539 | `experiments/final_evaluation/test_metrics.json` | **Verified** |
| **Pneumonia AUROC** | 0.8183 | 0.8183 | `experiments/final_evaluation/test_metrics.json` | **Verified** |
| **Atelectasis AUROC** | 0.8076 | 0.8076 | `experiments/final_evaluation/test_metrics.json` | **Verified** |
| **Pleural Effusion AUROC** | 0.9358 | 0.9358 | `experiments/final_evaluation/test_metrics.json` | **Verified** |
| **Pneumonia Sensitivity (threshold 0.5)** | 0.0 (0/8 detected) | 0.0 (0/8 detected) | Confirmed: 8 false negatives due to extreme class imbalance | **Verified** |
| **Track B Classical Head AUROC** | 0.8060 | 0.8060 | `experiments/qml/track_b_metrics.json` | **Verified** |
| **Track B Quantum Head AUROC** | 0.6488 | 0.6488 | `experiments/qml/track_b_metrics.json` | **Verified** |
| **Quantum Advantage Demonstrated?** | No | No (Classical +0.1572 higher) | `experiments/qml/qml_comparison_report.md` | **Verified** |
| **Quantum Implementation Type** | Simulation | Classical SO(2) simulation | `src/models/classical/qml.py` | **Verified** |
| **Clinical Diagnostic Readiness** | Research Prototype | Non-diagnostic draft | Clinical advisory disclaimer in UI and API | **Verified** |

---

## 6. Automated Test Suite Execution Results

Executed from project root using `.venv_cuda\Scripts\python.exe -m pytest tests/ --tb=short -q`:

- **Total Test Files:** 15
- **Total Tests:** 104
- **Passed:** 103
- **Skipped:** 1 (`tests/test_training_recovery.py::test_gpu_training_recovers_gracefully_from_oom` skipped because automated CI runner operates in CPU fallback mode for recovery test)
- **Failed:** 0
- **Execution Duration:** 31.28 seconds
- **Exit Code:** 0

### Test Breakdown by Subsystem:
- `tests/test_api.py`: 12 passed
- `tests/test_contracts.py`: 6 passed
- `tests/test_dataset_protocol.py`: 8 passed
- `tests/test_deterministic_preprocessing.py`: 8 passed
- `tests/test_full_training.py`: 6 passed
- `tests/test_phase2_modules.py`: 12 passed
- `tests/test_pipeline_leak_prevention.py`: 4 passed
- `tests/test_preprocessing.py`: 7 passed
- `tests/test_qml.py`: 6 passed
- `tests/test_qml_track_b.py`: 7 passed
- `tests/test_registry.py`: 4 passed
- `tests/test_report.py`: 7 passed
- `tests/test_settings.py`: 5 passed
- `tests/test_training_recovery.py`: 3 passed, 1 skipped
- `tests/test_xai.py`: 8 passed

---

## 7. EDI College Submission Checklist

Review of academic and report requirements for final college submission:

| Required Section | Status in Repository | What Exists | Action Required by User |
|:---|:---:|:---|:---|
| **1. Problem Definition & Objectives** | **Verified** | Documented in `README.md` and `experiments/project_complete_progress_report.md`. Multi-label screening on chest radiographs. | None (Ready for report export). |
| **2. Related Work & Literature Review** | **Requires User Input** | CheXpert benchmark references, DenseNet-121 baseline, and PennyLane QML background documented. | User must provide exact formal paper citations (Irvin et al. 2019, Huang et al. 2017) formatted for college guidelines. |
| **3. Proposed Solution & Methodology** | **Verified** | End-to-end pipeline: Preprocessing (224×224 grayscale, CLAHE, standardization), DenseNet-121 backbone, Grad-CAM, rule-based report generator. | None (Fully documented). |
| **4. System Architecture & Dataflow** | **Verified** | ASCII diagrams and dataflow specifications in `experiments/project_complete_progress_report.md` Section 4. | Include diagram in presentation slides. |
| **5. Dataset & Splitting Methodology** | **Verified** | CheXpert-small (172,033 train, 18,994 val, 202 frontal test). Patient-disjoint partitioning rigorously verified. | None (Fully documented). |
| **6. Results, Analysis & QML Comparison** | **Verified** | AUROC tables, bootstrap CIs ($B=2000$), sensitivity trade-off analysis, and Track B comparison table complete. | None (Use tables directly in report). |
| **7. Cost & Computational Sustainability** | **Requires User Input** | Training ran on NVIDIA GTX 1650 4GB GPU for ~5.6 hours. | User must input local power/hardware cost estimates if required by project rubric. |
| **8. Team Members & Responsibilities** | **Requires User Input** | Not present in code repository. | User must add student names, roll numbers, guide/mentor name, and individual module allocations. |
| **9. Demo Instructions & Future Work** | **Verified** | Exact startup scripts, test commands, and future directions (temperature scaling, LLM grounding) fully written. | Rehearse demo live before viva. |

---

## 8. Exact Commands to Launch Demo and Run Tests

### Launch Interactive Web Application (Recommended Configuration):
```powershell
# From project root: E:\College\QRadAssist

# Set checkpoint path to the 5-epoch full GPU model:
$env:QRADASSIST_CHECKPOINT_PATH = "experiments/full_gpu/best_full_model.pt"

# Start FastAPI server on localhost:
.venv_cuda\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
Open browser to: **`http://127.0.0.1:8000/`**

### Alternative Launch (Production Default Checkpoint):
```powershell
# Unset override to use src/models/classical/best_model.pt
$env:QRADASSIST_CHECKPOINT_PATH = ""
.venv_cuda\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### Run Entire Verification Test Suite:
```powershell
.venv_cuda\Scripts\python.exe -m pytest tests/ --tb=short -q
```

---

## 9. Known Limitations & Audit Boundaries

1. **Uncalibrated Decision Threshold for Pneumonia:** Due to high class imbalance in the training cohort, Pneumonia predictions rarely exceed $0.50$ (sensitivity $0.0$ at default cutoff). Operating threshold tuning (e.g., threshold sweep on validation set) is recommended before any prospective study.
2. **Rule-Based Draft Reporting:** The report generator (`src/models/classical/report.py`) uses deterministic sentence templates based on thresholded probabilities. It does not employ an LLM.
3. **Quantum Simulator Scope:** The QML component is an SO(2) rotation simulation in PyTorch on 2 simulated qubits. It was executed on classical hardware, not a physical quantum processor.
4. **Frontal View Assumption:** The primary classifier is designed for frontal (PA/AP) chest radiographs; lateral radiographs are not processed by the primary evaluation pipeline.
5. **Academic/Research Scope:** The application is not certified under FDA/CE medical device software regulations and is strictly an educational research prototype.
