# QRadAssist Phase 3 — Explainability and Demo Validation Report

**Validation Timestamp (UTC):** 2026-10-21T15:17:00Z  
**Evaluated Model Checkpoint:** `experiments/full_gpu/best_full_model.pt`  
**Checkpoint SHA-256:** `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a`  
**Production Checkpoint:** `src/models/classical/best_model.pt`  
**Production SHA-256:** `392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a`  

---

## 1. Components Inspected & Minimal Changes Made

| Component / File | Purpose | Changes Applied |
| :--- | :--- | :--- |
| `src/models/classical/xai.py` | Grad-CAM explainability generation | Replaced deprecated module `register_backward_hook` with tensor `output.register_hook` registered during the forward pass. This resolves the PyTorch `FutureWarning` and avoids in-place ReLU view conflicts in `torchvision` DenseNet-121 while maintaining identical mathematical output. |
| `src/api/app.py` | FastAPI backend service | Added `_get_checkpoint_path()` to support runtime checkpoint override via `QRADASSIST_CHECKPOINT_PATH` environment variable while strictly defaulting to production `CHECKPOINT_PATH`. Added support for both raw state dicts and epoch-dict structures. |
| `src/frontend/script.js` | Web interface logic | Enhanced report renderer to display prominent clinical advisory banners: `⚠️ AI-GENERATED DRAFT REPORT — REQUIRES QUALIFIED CLINICIAN REVIEW` and verification notices. Confirmed sigmoid is applied exactly once to raw model logits. |
| `src/frontend/index.html` | Web presentation structure | Added distinct clinical advisory badges to the draft report section and explainability notes to the Grad-CAM visualization section. |
| `tests/test_xai.py` | Explainability test suite | Added `test_gradcam_class_differentiation` and `test_gradcam_trained_checkpoint_integration` to verify pathology-specific attribution and real checkpoint loading. Fixed regex raw-string escaping. |
| `tests/test_api.py` | API and integration tests | Added `test_checkpoint_path_resolution_env_override` and `test_end_to_end_workflow_experimental_model` verifying complete image upload → predict → report → Grad-CAM flow using `best_full_model.pt`. |

---

## 2. Grad-CAM Implementation & Validation Results

### 2.1 Convolutional Feature Layer Targeting
* **Target Layer:** `model.backbone.features[-1]` (`norm5`, BatchNorm2d with 1,024 channels), which is the final feature representation before global pooling and classification in DenseNet-121.
* **Attribution Target:** The unnormalized class logit `score = logits[0, target_class]`.
* **Output Dimensions:** Bounded 2D heatmap of shape `(224, 224)` matching the preprocessed image resolution.

### 2.2 Numerical & Differentiation Properties
1. **Range and Finiteness:**
   - All heatmap values are strictly finite (zero NaNs, zero Infs).
   - Values are strictly bounded in $[0.0, 1.0]$.
   - When feature activations have non-positive gradient attribution, the function cleanly outputs finite zeros.
2. **Pathology-Specific Spatial Differentiation:**
   - On identical input tensors, Grad-CAM produces distinct spatial activation maps for different target conditions:
     - Mean absolute difference between Pneumonia and Atelectasis heatmaps: **0.2206**
     - Mean absolute difference between Pneumonia and Pleural Effusion heatmaps: **0.2230**
     - Confirmed via automated regression test `test_gradcam_class_differentiation`.
3. **Execution on Trained Experimental Checkpoint:**
   - Evaluated on `experiments/full_gpu/best_full_model.pt`.
   - On synthetic inputs and held-out test benchmarks, generates localized heatmaps (e.g., active localization with mean `0.2673` on `patient64570`).

> [!CAUTION]
> **Clinical Validity Guardrail:** Grad-CAM heatmaps highlight image regions that positively contribute to intermediate feature activations for a specific class score. **Grad-CAM does not prove pathological presence, model correctness, or clinical validity.** Heatmaps must not be interpreted as anatomical segmentations or diagnostic boundaries.

---

## 3. End-to-End API-to-Frontend Data-Flow Verification

The end-to-end workflow was tested using the FastAPI `TestClient` with `QRADASSIST_CHECKPOINT_PATH=experiments/full_gpu/best_full_model.pt`:

```
Uploaded Chest X-ray (JPEG)
           │
           ▼
Deterministic Preprocessor ──► (1, 1, 224, 224) Float32 Tensor [0.0, 1.0]
           │
           ├─────────────────────────────┬─────────────────────────────┐
           ▼                             ▼                             ▼
    POST /predict                   POST /report                 POST /gradcam
           │                             │                             │
   Raw Logits (1, 3)             Sigmoid Probabilities         Target Class Score
           │                             │                             │
    JSON Probabilities          generate_report()            Backward Autograd Hook
  {"probabilities": ...}                 │                             │
           │                     JSON Draft Report             Normalized 2D Heatmap
           ▼                  {"findings", "impression"}        [[float, ...], ...]
   Frontend Script                       │                             │
           │                             ▼                             ▼
   Single Sigmoid Applied        Prominent Clinical             Overlay Canvas on
  Display % to User             Draft Disclaimer Display        Original X-ray
```

### 3.1 Sigmoid Application Audit
* **`/predict` Endpoint:** Returns raw model logits under the `"probabilities"` key to preserve backward compatibility with legacy client tests (`test_predict_success`). The frontend (`script.js` line 74) evaluates `(v < 0 || v > 1) ? 1 / (1 + Math.exp(-v)) : v;`, ensuring sigmoid is applied **exactly once**.
* **`/report` Endpoint:** The backend explicitly executes `torch.sigmoid(logits)` **exactly once** before threshold comparison (`threshold=0.5`) in `generate_report()`. The frontend displays the resulting findings directly without further transformation.
* **`/gradcam` Endpoint:** Operates directly on unnormalized scalar class scores `logits[0, target_class]` without sigmoid compression, adhering to standard Grad-CAM formulation.

### 3.2 Schema Consistency
All API responses adhere strictly to defined Pydantic and JSON structures:
* `/health` $\to$ `{"status": "ok"}`
* `/labels` $\to$ `["Pneumonia", "Atelectasis", "Pleural Effusion"]`
* `/predict` $\to$ `{"probabilities": {"Pneumonia": float, "Atelectasis": float, "Pleural Effusion": float}}`
* `/report` $\to$ `{"findings": list[str], "impression": str, "heatmaps": dict}`
* `/gradcam` $\to$ `{"heatmap": list[list[float]]}` (224×224 array)

### 3.3 Draft Report Clinical Framing
All reports produced in the user interface prominently display the following advisory:
```
⚠️ AI-GENERATED DRAFT REPORT — REQUIRES QUALIFIED CLINICIAN REVIEW
NOT A DEFINITIVE DIAGNOSIS — FOR CLINICAL DECISION SUPPORT ONLY

=== Findings ===
...
=== Impression ===
...
=== Verification Notice ===
Preliminary automated assessment. Review by a qualified radiologist or physician required.
```

---

## 4. Error Handling Verification

The API was tested for graceful error handling across failure modes:
1. **Corrupted / Non-Image Uploads:** Uploading plain text or invalid image bytes to `/predict`, `/report`, or `/gradcam` returns **HTTP 422 Unprocessable Content** with detail message `"Uploaded file must be an image."` or `"Uploaded file is not a valid image."` (no 500 errors, no unhandled stack traces).
2. **Unrecognized Target Disease:** Submitting an invalid pathology string to `/gradcam` returns **HTTP 422 Unprocessable Content** with detail message `"Disease '...' is not a recognized target label."`.
3. **Missing / Unresolvable Checkpoint:** If the configured checkpoint file is missing or deleted, the service returns **HTTP 503 Service Unavailable** with detail message `"Model checkpoint not found; inference unavailable."`.

---

## 5. Test Suite Execution & Outcomes

Executed the complete regression suite using the virtual environment `.venv_cuda`:

```powershell
.venv_cuda\Scripts\python.exe -m pytest tests/test_xai.py tests/test_report.py tests/test_api.py tests/test_training_recovery.py tests/test_phase2_modules.py -v --tb=short
```

* **Test Suite Status:** **38 passed, 1 skipped, 0 failed**
* **Exit Code:** `0`
* **Test Breakdown:**
  - `tests/test_xai.py`: 5 passed (shape, invalid batch, invalid index, class differentiation, trained checkpoint integration)
  - `tests/test_report.py`: 4 passed (basic report, no findings, invalid shape, heatmaps attachment)
  - `tests/test_api.py`: 10 passed (health, predict, invalid image, missing checkpoint, report, gradcam success, invalid disease, invalid image, env override, end-to-end integration)
  - `tests/test_training_recovery.py`: 15 passed, 1 skipped (device, atomic save, recovery, checkpoint isolation)
  - `tests/test_phase2_modules.py`: 4 passed (dataset loading, loss masking, metrics edge cases, output dimensions)

---

## 6. Checkpoint & Artifact Integrity Audit

* **Production Checkpoint (`src/models/classical/best_model.pt`):**
  - Expected SHA-256: `392B65C9130717BFB19053F5E638EF31BB9FCDDC40E169399DBC89CB15B9D25A`
  - Current SHA-256:  `392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a`
  - Status: **100% UNCHANGED AND UNTOUCHED**.
* **Experimental Checkpoint (`experiments/full_gpu/best_full_model.pt`):**
  - Current SHA-256: `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a`
  - Status: **100% UNCHANGED**.
* **Phase 2 Evaluation Artifacts (`experiments/final_evaluation/`):**
  - `test_metrics.json`: Unchanged (6,691 bytes)
  - `test_metrics.csv`: Unchanged (1,072 bytes)
  - `evaluation_report.md`: Unchanged (5,933 bytes)
  - `evaluation_audit.md`: Unchanged (9,699 bytes)

---

## 7. Known Limitations & Remaining Risks

1. **Pneumonia Zero Sensitivity at 0.5 Threshold:** As audited in Phase 2, the model's raw uncalibrated probabilities for pneumonia fall below 0.5 on held-out positives due to low training prevalence (2.17%). While the ranking capability is strong (AUROC 0.8183), default 0.5 thresholding yields zero positive detections for pneumonia in draft reports. Calibrated operating points (e.g. 80% sensitivity) must be selected before clinical evaluation.
2. **Grad-CAM Coarse Spatial Resolution:** DenseNet-121's feature map before pooling has a spatial resolution of 7×7. Upsampling 7×7 maps to 224×224 via bilinear interpolation provides smooth regional localization, but cannot delineate fine sub-centimeter lung opacities or precise lesion contours.
3. **No External Multi-Center Generalization:** The model and explainability have only been validated on the single-institution Stanford CheXpert dataset. Domain shifts from different scanner manufacturers, pediatric populations, or ICU bedside positioning remain unmeasured.
