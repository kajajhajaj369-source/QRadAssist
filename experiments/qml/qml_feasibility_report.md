# QRadAssist — Hybrid QML Feasibility & Experiment Audit Report

**Audit Date:** 2026-10-21  
**Target Architecture:** Hybrid Quantum-Classical Classifier Branch  
**Target Milestone:** Research Question 3 (RQ3) / Original Roadmap Phase 4  
**Audit Purpose:** Pre-experimental inspection, dependency audit, hardware validation, dataset verification, and comparative experiment design.  
**Operating Constraint:** Audit and planning only. No source code modifications, model retraining, or checkpoint changes performed.

---

> [!IMPORTANT]
> **Summary Verdict:**  
> 1. The current QML implementation in [`src/models/classical/qml.py`](file:///E:/College/QRadAssist/src/models/classical/qml.py) is a **classical PyTorch simulation stub** using trigonometric rotation matrices on a 2-qubit statevector. It does **not** execute on a quantum computer (QPU), nor does it use a full QML framework (such as PennyLane, Qiskit, or TorchQuantum).
> 2. The unit test suite [`tests/test_qml.py`](file:///E:/College/QRadAssist/tests/test_qml.py) passes with **100% success (4/4 tests passed, exit code 0)**.
> 3. The development cohort [`data/splits/dev_cohort_manifest.csv`](file:///E:/College/QRadAssist/data/splits/dev_cohort_manifest.csv) exists (15,000 frontal radiographs across 5,084 patients) and has been programmatically verified to have **strict zero patient overlap** with both the internal validation set and the held-out test split.
> 4. Neither PennyLane nor Qiskit are installed in `.venv_cuda`. However, the current PyTorch-native implementation is fully compatible with the existing CUDA environment (GTX 1650, 4.0 GB VRAM) and can be trained immediately without external dependencies.
> 5. A fair, rigorous two-stage comparison protocol against DenseNet-121 is designed below.

---

## 1. Step 1 — Inspection of Existing Implementation & Architecture

### 1.1 Source Code & Module Inspection

1. **[`src/models/classical/qml.py`](file:///E:/College/QRadAssist/src/models/classical/qml.py):**
   - **Classes Defined:** `SimpleQuantumLayer` and `HybridQuantumClassifier`.
   - **Encoder:** `nn.Flatten()` followed by `nn.Linear(1 * 224 * 224, n_qubits)`. For $n\_qubits=2$, this layer contains $50,176 \times 2 + 2 = 100,354$ parameters, projecting the raw $224 \times 224$ grayscale image directly to 2 real-valued angles ($\theta_0, \theta_1$).
   - **"Quantum" Layer (`SimpleQuantumLayer`):**
     - Contains a learnable 1D parameter tensor `self.params = nn.Parameter(torch.randn(n_qubits))` (2 parameters).
     - Computes single-qubit $SO(2)$ rotation matrices for each qubit:
       $$R_i = \begin{pmatrix} \cos(\theta_i + p_i) & -\sin(\theta_i + p_i) \\ \sin(\theta_i + p_i) & \cos(\theta_i + p_i) \end{pmatrix}$$
     - Initializes the state vector $|00\rangle = (1, 0, 0, 0)^T$.
     - Computes sequential matrix multiplications via `torch.einsum('bik,bkj->bij', rot, state)` to produce a batch of $2^{n\_qubits}$-dimensional statevectors (dimension 4 for 2 qubits).
   - **Classifier Head:** `nn.Linear(2**n_qubits, len(TARGET_LABELS))` ($4 \to 3$ linear head, 15 parameters).
   - **Total Parameter Count:** Exactly **100,371 parameters** (100,354 in encoder, 2 in quantum layer, 15 in classification head).
   - **Label Contract:** Preserves multi-label ordering matching `preprocessing.labels.TARGET_LABELS` (`Pneumonia`, `Atelectasis`, `Pleural Effusion`). Outputs raw logits compatible with `masked_bce_with_logits`.
   - **Channel Handling:** Accepts single-channel $(B, 1, 224, 224)$ or three-channel $(B, 3, 224, 224)$ inputs (slices channel 0 if 3 channels provided).

2. **Nature of the Implementation (Genuine vs. Simulator vs. Classical):**
   - **Explicit Finding:** This is **strictly a classical PyTorch numerical simulation**.
   - As documented in line 14 of `qml.py`:
     > *"It is not a true quantum simulation – it is a deterministic, fully-classical computation that preserves the API shape of a quantum circuit for experimental purposes."*
   - There are **no quantum gates, no circuit transpilation, no shot noise / measurement sampling, and no quantum hardware backends**.
   - **Quantum Advantage Claim:** Under no circumstances can quantum advantage be claimed from this implementation. Any report or publication must explicitly describe this as a *classical parameterized rotation circuit simulation*.

3. **[`src/models/qml/__init__.py`](file:///E:/College/QRadAssist/src/models/qml/__init__.py):**
   - Package docstring only (7 lines):
     > *"Experimental hybrid quantum-classical / QML branch. Intended for later comparison with a classical approach. Presence of this package does not imply quantum advantage. Implementation is deferred past Phase 0."*
   - Contains no executable classes; all actual implementation currently resides in `src/models/classical/qml.py`.

4. **Integration with Contracts & Registry:**
   - [`src/contracts.py`](file:///E:/College/QRadAssist/src/contracts.py) defines `DiseasePrediction` and `AnalysisFindings`, which are model-agnostic. The output logits from `HybridQuantumClassifier` convert to calibrated probabilities via sigmoid, fulfilling the pipeline contract.
   - [`src/registry.py`](file:///E:/College/QRadAssist/src/registry.py) defines `DiseaseRegistry` with 3 active labels (`pneumonia`, `atelectasis`, `pleural_effusion`), matching the 3 output dimensions of `HybridQuantumClassifier`.

---

## 2. Step 2 — Experimental Feasibility & Dependency Audit

### 2.1 Unit Test Execution

Executed the existing QML test suite using the project environment:
```powershell
.venv_cuda\Scripts\python.exe -m pytest tests/test_qml.py -v --tb=short
```

**Results:**
```
tests/test_qml.py::test_hybrid_qml_forward_shape PASSED                  [ 25%]
tests/test_qml.py::test_hybrid_qml_forward_three_channels PASSED         [ 50%]
tests/test_qml.py::test_hybrid_qml_gradient_flow PASSED                  [ 75%]
tests/test_qml.py::test_hybrid_qml_deterministic_output PASSED           [100%]

============================== 4 passed in 3.60s ==============================
Exit Code: 0
```
- **Forward Shape:** Correctly outputs $(1, 3)$ logits for $(1, 1, 224, 224)$ input.
- **Three-Channel Support:** Correctly processes $(1, 3, 224, 224)$ input.
- **Gradient Flow:** Backward pass propagates non-None gradients to all parameters (`encoder.1.weight`, `encoder.1.bias`, `quantum.params`, `classifier.weight`, `classifier.bias`).
- **Determinism:** Identical inputs yield identical outputs under fixed seed.

### 2.2 Dependency & Environment Audit

- **Python Runtime:** Python 3.14.2 (via `.venv_cuda\Scripts\python.exe`).
- **Core Frameworks:**
  - `torch`: `2.15.0.dev20260907+cu126` (CUDA 12.6 supported).
  - `torchvision`: `0.30.0.dev20260906+cu126`.
- **Specialized QML Frameworks:**
  - `pennylane`: **Not Installed**
  - `qiskit`: **Not Installed**
  - `torchquantum`: **Not Installed**
  - `cirq`: **Not Installed**
- **Feasibility Implication:**
  - Because `src/models/classical/qml.py` is written entirely in native PyTorch (`torch.nn.Module`, `torch.einsum`, `torch.cos`, `torch.sin`), **no additional packages need to be installed** to train or evaluate `HybridQuantumClassifier`.
  - It runs out-of-the-box on both CPU and CUDA without dependency conflicts.

### 2.3 Hardware & Compute Resources

- **GPU Device:** NVIDIA GeForce GTX 1650
- **Total Dedicated VRAM:** 4.0 GB (4,096 MiB)
- **Host RAM:** 16 GB
- **Throughput & Footprint:**
  - DenseNet-121 has ~7.0 million parameters and required ~1.8 GB VRAM at batch size 8.
  - `HybridQuantumClassifier` has only **100,371 parameters** (~0.4 MB model weight size).
  - Estimated VRAM footprint at batch size 16: **< 350 MiB**.
  - Estimated training throughput: **~120–150 samples/second** on GTX 1650 (compared to ~25 samples/s for DenseNet-121).

---

## 3. Step 3 — Dataset & Cohort Separation Audit

### 3.1 Partition Verification & Overlap Analysis

An exhaustive patient ID set intersection was performed across all partitions:

| Partition | File Path | Total Rows | Unique Patients | Leakage into Other Sets |
|:---|:---|:---:|:---:|:---:|
| **Dev Cohort** | `data/splits/dev_cohort_manifest.csv` | 15,000 | 5,084 | **0 with Val, 0 with Test** (100% subset of Train) |
| **Full Train** | `data/splits/train_manifest.csv` | 172,033 | 58,081 | **0 with Val, 0 with Test** |
| **Internal Val** | `data/splits/val_manifest.csv` | 18,994 | 6,453 | **0 with Train, 0 with Test** |
| **Held-Out Test** | `data/raw/chexpert-small/valid.csv` (Frontal) | 202 | 200 | **0 with Train, 0 with Val, 0 with Dev** |

**Confirmed Facts:**
1. `dev_cohort_manifest.csv` exists and is 100% derived from `train_manifest.csv`.
2. It contains whole patients (all frontal studies of all 5,084 patients are included).
3. It has **strict zero patient overlap** with `val_manifest.csv` and `data/raw/chexpert-small/valid.csv`.
4. It is fully suitable for training development models without risking data leakage into validation or held-out test splits.

### 3.2 Label Distribution in the Development Cohort ($N = 15,000$)

| Target Disease | Positive (`1.0`) | Negative (`0.0`) | Uncertain (`-1.0`) | Blank (Unmentioned) |
|:---|:---:|:---:|:---:|:---:|
| **Pneumonia** | 272 (1.81%) | 185 (1.23%) | 1,358 (9.05%) | 13,185 (87.90%) |
| **Atelectasis** | 2,391 (15.94%) | 31 (0.21%) | 2,471 (16.47%) | 10,107 (67.38%) |
| **Pleural Effusion** | 6,460 (43.07%) | 1,938 (12.92%) | 501 (3.34%) | 6,101 (40.67%) |

**Clinical Prevalence Observations:**
- Under the project's standard `U-Ignore` policy:
  - Blank is imputed as negative ($0.0$, weight $1.0$).
  - Uncertain (`-1.0`) is masked out (weight $0.0$).
  - Effective training counts in Dev Cohort:
    - **Pneumonia:** 272 positives, 13,370 negatives, 1,358 masked. (Severe imbalance: 1.81% positive).
    - **Atelectasis:** 2,391 positives, 10,138 negatives, 2,471 masked. (Moderate imbalance: 15.94% positive).
    - **Pleural Effusion:** 6,460 positives, 8,039 negatives, 501 masked. (Balanced: 43.07% positive).
- Every target label has adequate positive and negative samples in the dev cohort; AUROC calculation is well-defined.

---

## 4. Step 4 — Comparative Evaluation Design (DenseNet-121 vs. Hybrid QML)

### 4.1 Key Architectural Observation & Fair Comparison Challenge

A critical architectural distinction must be documented:
1. **DenseNet-121:**
   - 121 layers deep, convolutional feature extractor, 7M parameters.
   - Learns hierarchical spatial features (edges $\to$ textures $\to$ anatomical structures $\to$ lesions).
2. **Current `HybridQuantumClassifier`:**
   - Uses a **single linear layer** from $50,176$ raw pixels directly to 2 angles.
   - It possesses **no convolutional filters** and cannot learn localized spatial radiological features.
   - A direct end-to-end comparison of raw pixel `HybridQuantumClassifier` against DenseNet-121 will inevitably show that DenseNet-121 vastly outperforms it. This would primarily measure the difference between a 121-layer CNN vs. a 1-layer linear encoder, rather than isolating the contribution of the quantum layer.

### 4.2 Proposed Two-Track Comparative Protocol

To answer Research Question 3 (*"How does a hybrid quantum-classical classifier compare with a classical classifier for the selected chest X-ray classification task?"*) with scientific validity, we propose a two-track experiment:

#### Track A: Direct Baseline Evaluation (As-Is Architecture)
- **Model:** `HybridQuantumClassifier(n_qubits=2)` trained end-to-end on raw pixels.
- **Dataset:** 5,000 deterministic samples from `data/splits/train_manifest.csv` (identical to the Phase 9 dev run in `experiments/run_dev_training.py`).
- **Validation Subset:** 1,500 deterministic samples from `data/splits/val_manifest.csv` (`experiments/dev_gpu/val_dev_1500.csv`).
- **Loss & Policy:** `masked_bce_with_logits` under `u_ignore` policy.
- **Epochs:** 3 epochs, batch size 16, learning rate $1\times 10^{-3}$, Adam optimizer.
- **Reference Baseline:** Compare directly with DenseNet-121 dev model (`experiments/dev_gpu/best_dev_model.pt`) which was trained on the exact same 5,000 training samples and evaluated on the exact same 1,500 validation samples.

#### Track B: Feature-Extracted Head Comparison (True Quantum vs. Classical Head)
- **Motivation:** In modern hybrid QML (and as originally diagrammed in `docs/architecture/overview.md` lines 15–17):
  $$\text{Image} \longrightarrow \text{DenseNet-121 Backbone} \longrightarrow \text{Feature Vector } (d=1024) \longrightarrow \text{Quantum / Classical Head}$$
- **Protocol:**
  - Extract the 1024-dimensional feature representations from the trained DenseNet-121 (`experiments/full_gpu/best_full_model.pt`).
  - Train Head 1: **Classical Linear Head** ($1024 \to 3$).
  - Train Head 2: **Hybrid Quantum Head** ($1024 \to \text{Linear to } 2 \text{ or } 4 \text{ qubits} \to \text{SimpleQuantumLayer} \to \text{Linear to } 3$).
  - Compare both heads on identical frozen representations. This isolates the mathematical expressive power of the parameterized unitary from the feature extraction burden.

### 4.3 Evaluation Metrics

To ensure strict comparability with Phase 10 evaluation reports:
- **Primary Metric:** Macro-average AUROC across `Pneumonia`, `Atelectasis`, and `Pleural Effusion`.
- **Secondary Metric:** Macro-average Average Precision (AP).
- **Per-Class Metrics:**
  - AUROC and AP for each individual condition.
  - Number of positive, negative, and masked samples reported.
- **Held-Out Test Set Isolation:**
  - The held-out benchmark (`data/raw/chexpert-small/valid.csv`, $N=202$) will **never** be seen during training, validation, or checkpoint selection.
  - Final inference on the held-out benchmark will only be executed once models are locked.

---

## 5. Step 5 — Estimated Compute Requirements & Time Budget

| Phase / Run | Training Samples | Batch Size | Epochs | Est. VRAM | Est. Runtime | Destination Directory |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Track A Dev Run** | 5,000 | 16 | 3 | < 400 MiB | ~2–4 minutes | `experiments/qml/track_a/` |
| **Track A Full Dev** | 15,000 | 16 | 5 | < 400 MiB | ~8–12 minutes | `experiments/qml/track_a_full/` |
| **Track B Head Run** | 15,000 (pre-extracted) | 64 | 10 | < 100 MiB | < 1 minute | `experiments/qml/track_b/` |

**Hardware Safety Guarantee:**
- The GTX 1650 (4 GB VRAM) has more than 8x the required memory headroom for this experiment.
- Out-of-memory (OOM) risk is negligible.

---

## 6. Protection of Existing Project Artifacts

The integrity of all existing models and reports has been independently verified:
- **Production Baseline Checkpoint:** `src/models/classical/best_model.pt`  
  - SHA-256: `392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a` (**Verified untouched**).
- **Experimental Full GPU Checkpoint:** `experiments/full_gpu/best_full_model.pt`  
  - SHA-256: `4b74019200acc0d0e7a9cdc7d5e005a6fb31ad55a482e947759808fd9726315a` (**Verified untouched**).
- **Evaluation Reports:** `experiments/final_evaluation/` and `experiments/phase3/` remain intact and unedited.

---

## 7. Concrete Next Implementation Steps (Awaiting User Instruction)

When instructed to proceed with execution, the exact operational steps will be:

1. **Create isolated experiment directory:** `experiments/qml/` to receive all logs, metrics, and checkpoints without contaminating production or previous experimental directories.
2. **Implement training driver:** `experiments/qml/train_qml.py` following the existing `experiments/run_dev_training.py` architecture (atomic saving, logging, masked BCE loss, and AUROC calculation).
3. **Execute Track A Training:** Train `HybridQuantumClassifier` on the 5,000-sample development split and evaluate on `experiments/dev_gpu/val_dev_1500.csv` to establish direct comparison with the DenseNet-121 development model.
4. **Evaluate on Held-Out Test Set:** Run inference on the 202 frontal images in `data/raw/chexpert-small/valid.csv`.
5. **Generate Comparison Report:** Emit `experiments/qml/qml_comparison_report.md` documenting per-label AUROC, AP, training time, and honest clinical/computational conclusions regarding classical vs. hybrid simulation performance.
