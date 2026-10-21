# QRadAssist — Track B QML Experiment: Classical vs. Simulated Quantum Head Comparison Report

**Experiment Date:** 2026-10-21  
**Experiment Name:** Track B Feature-Grounded Multi-Label Classification  
**Target Architecture:** Frozen DenseNet-121 Feature Extractor + Trainable Classification Heads  
**Research Question Addressed:** **RQ3** (*"How does a hybrid quantum-classical classifier compare with a classical classifier for the selected chest X-ray classification task?"*)  
**Execution Environment:** CUDA (NVIDIA GeForce GTX 1650, 4.0 GB VRAM, Windows 11)  
**Deliverable Artifacts:**
- Experiment Runner: [`experiments/qml/run_track_b.py`](file:///E:/College/QRadAssist/experiments/qml/run_track_b.py)
- Head Architectures: [`experiments/qml/qml_heads.py`](file:///E:/College/QRadAssist/experiments/qml/qml_heads.py)
- Feature Extractor Utility: [`experiments/qml/feature_extractor.py`](file:///E:/College/QRadAssist/experiments/qml/feature_extractor.py)
- Unit Tests: [`tests/test_qml_track_b.py`](file:///E:/College/QRadAssist/tests/test_qml_track_b.py)
- Metrics Payload: [`experiments/qml/track_b_metrics.json`](file:///E:/College/QRadAssist/experiments/qml/track_b_metrics.json)
- Checkpoints: `experiments/qml/best_classical_head.pt`, `experiments/qml/best_quantum_sim_head.pt`

---

> [!IMPORTANT]
> **Scientific Integrity & Nature of Implementation:**
> 1. The quantum head evaluated in this experiment utilizes the existing `SimpleQuantumLayer` from [`src/models/classical/qml.py`](file:///E:/College/QRadAssist/src/models/classical/qml.py).
> 2. This implementation is a **classical PyTorch simulation** calculating 2D trigonometric rotation matrices on a 2-qubit statevector. It does **not** execute on a quantum computer (QPU), nor does it simulate quantum noise, circuit depth, or measurement collapse.
> 3. **No quantum advantage is claimed or demonstrated.** As shown empirically below, classical linear classification significantly outperforms the 2-qubit simulated quantum head across all metrics.

---

## 1. Experimental Design & Protocol

### 1.1 Objective
To conduct a fair, head-to-head comparison isolating the expressive capacity of the simulated quantum layer from the burden of raw spatial image representation.

Both classification heads operated on identical, frozen 1024-dimensional feature embeddings extracted from the pre-trained full model:
- **Frozen Feature Extractor:** `experiments/full_gpu/best_full_model.pt` (`DenseNet121MultiLabel`). All weights frozen (`requires_grad=False`, `eval()` mode).
- **Head 1 (Classical Linear Head):**
  $$\mathbf{x} \in \mathbb{R}^{1024} \xrightarrow{\text{Linear}(1024, 3)} \mathbf{z} \in \mathbb{R}^3$$
  - Parameters: $1024 \times 3 + 3 = \mathbf{3,075}$ parameters.
- **Head 2 (Simulated Quantum Head, 2 Qubits):**
  $$\mathbf{x} \in \mathbb{R}^{1024} \xrightarrow{\text{Angle Encoder}(1024, 2)} \boldsymbol{\theta} \in \mathbb{R}^2 \xrightarrow{\text{SimpleQuantumLayer}} |\psi\rangle \in \mathbb{R}^4 \xrightarrow{\text{Linear}(4, 3)} \mathbf{z} \in \mathbb{R}^3$$
  - Parameters: Angle encoder ($1024 \times 2 + 2 = 2,050$), `SimpleQuantumLayer` ($2$ learnable phase shifts), Classifier ($4 \times 3 + 3 = 15$). Total: $\mathbf{2,067}$ parameters.

### 1.2 Data Partitions & Integrity
- **Training Subset:** First 5,000 deterministic samples from `data/splits/train_manifest.csv` (identical to the Phase 9 development training subset).
- **Validation Subset:** First 1,500 deterministic samples from `data/splits/val_manifest.csv` (`experiments/dev_gpu/val_dev_1500.csv`).
- **Patient Leakage:** Zero patient overlap between training and validation partitions.
- **Held-Out Test Set Isolation:** The designated held-out benchmark (`data/raw/chexpert-small/valid.csv`, $N=202$) was **strictly excluded** from training, feature caching, and model selection.
- **Loss Function:** `masked_bce_with_logits` under the `u_ignore` policy (uncertain `-1.0` labels masked out with weight $0.0$).
- **Hyperparameters:** Batch size 64, Adam optimizer, learning rate $1\times 10^{-3}$, weight decay $1\times 10^{-4}$, 15 epochs, random seed 42.

---

## 2. Pre- and Post-Experiment Checkpoint Integrity

Both existing model checkpoints were verified via SHA-256 before and after execution:

| Checkpoint Path | Expected SHA-256 | Pre-Run SHA-256 | Post-Run SHA-256 | Status |
|:---|:---:|:---:|:---:|:---:|
| `experiments/full_gpu/best_full_model.pt` | `4b74019200acc0...` | `4b74019200acc0...` | `4b74019200acc0...` | **100% UNTOUCHED** |
| `src/models/classical/best_model.pt` | `392b65c9130717...` | `392b65c9130717...` | `392b65c9130717...` | **100% UNTOUCHED** |

---

## 3. Validation Sample Counts & Label Distribution

Evaluated on $N = 1,500$ validation radiographs across 3 target labels:

| Target Disease | Unmasked Samples | Positive Count | Negative Count | Masked (Uncertain) | Pos Prevalence |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Pneumonia** | 1,364 | 22 | 1,342 | 136 | 1.61% |
| **Atelectasis** | 1,260 | 190 | 1,070 | 240 | 15.08% |
| **Pleural Effusion** | 1,442 | 620 | 822 | 58 | 43.00% |

All three labels contain positive and negative examples; zero AUROC metrics were undefined.

---

## 4. Empirical Results & Performance Comparison

Models were selected at their epoch of highest mean validation AUROC:
- **Classical Linear Head:** Best Epoch 3 (Train Loss: 0.2834, Val Loss: 0.2949)
- **Simulated Quantum Head:** Best Epoch 13 (Train Loss: 0.3921, Val Loss: 0.3883)

### 4.1 Comparative Metrics Table

| Target Condition | Classical Linear Head AUROC | Simulated Quantum Head AUROC | AUROC Delta ($\Delta_{\text{Q} - \text{C}}$) | Classical AP | Quantum Sim AP | AP Delta ($\Delta_{\text{Q} - \text{C}}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pneumonia** | **0.8176** | 0.4504 | $-0.3671$ | **0.1158** | 0.0141 | $-0.1018$ |
| **Atelectasis** | **0.7020** | 0.6055 | $-0.0965$ | **0.2733** | 0.2042 | $-0.0691$ |
| **Pleural Effusion** | **0.8986** | 0.8906 | $-0.0079$ | **0.8517** | 0.8493 | $-0.0024$ |
| **MACRO AVERAGE** | **0.8060** | **0.6488** | **$-0.1572$** | **0.4136** | **0.3559** | **$-0.0577$** |

### 4.2 Runtime and Resource Footprint

| Metric | Classical Linear Head | Simulated Quantum Head (2 Qubits) |
|:---|:---:|:---:|
| **Training Duration (15 epochs)** | **3.03 seconds** | **5.64 seconds** |
| **Inference Latency (1,500 samples)** | < 5 ms | < 12 ms |
| **Peak GPU Memory Allocated** | 191.3 MiB | 191.3 MiB |
| **Total Head Parameters** | 3,075 | 2,067 |

---

## 5. Scientific Analysis & Discussion

### 5.1 Why the Classical Head Dominates ($\Delta \text{AUROC} = +0.1572$)
1. **The 2-Qubit Dimensional Bottleneck:**
   The angle encoder forces the rich 1024-dimensional DenseNet feature space into just **2 scalar rotation angles** ($\theta_0, \theta_1$). Compressing 1024 semantic dimensions down to 2 continuous variables loses over 99.8% of the linear subspace information before any quantum rotation is computed.
2. **Catastrophic Failure on Low-Prevalence Pathology (Pneumonia: 0.4504 AUROC):**
   Pneumonia is the rarest condition in this cohort (1.61% positive prevalence). The 2-angle bottleneck discarded the subtle feature distinctions necessary to separate the 22 positive cases from the 1,342 negative cases, reducing Pneumonia performance to worse than random guessing (0.4504).
3. **Resilience on High-Prevalence Pathology (Pleural Effusion: 0.8906 AUROC):**
   Pleural Effusion is the most visually distinct and prevalent condition (43.0% positive). Even under extreme 2-angle compression, the dominant principal component of the 1024-d embedding correlates strongly with effusion, allowing the simulated quantum head to achieve 0.8906 AUROC (only $0.0079$ below the classical head).
4. **Optimization Trajectory:**
   The classical head converged almost immediately (Epoch 3, Val Loss 0.2949), whereas the non-linear sinusoidal loss landscape of `SimpleQuantumLayer` required 13 epochs to settle into an inferior minimum (Val Loss 0.3883).

---

## 6. Unit & Regression Test Verification

### 6.1 Focused Track B Tests
Command: `.venv_cuda\Scripts\python.exe -m pytest tests/test_qml_track_b.py tests/test_qml.py -v --tb=short`
- `test_feature_extractor_output_shape`: **PASSED**
- `test_classical_linear_head_shape_and_grad`: **PASSED**
- `test_simulated_quantum_head_shape_and_grad`: **PASSED**
- `test_masked_loss_with_uncertain_labels`: **PASSED**
- `test_heads_reproducibility`: **PASSED**
- `test_evaluation_metrics_computation`: **PASSED**
- `test_hybrid_qml_forward_shape`: **PASSED**
- `test_hybrid_qml_forward_three_channels`: **PASSED**
- `test_hybrid_qml_gradient_flow`: **PASSED**
- `test_hybrid_qml_deterministic_output`: **PASSED**
- **Result:** **10 passed in 5.19s (Exit Code: 0)**

---

## 7. Limitations & Honest Clinical Conclusions

1. **No True Quantum Circuitry:** The simulation used classical PyTorch math ($SO(2)$ matrix multiplications on statevectors). No entanglement gates ($CX$), variational Ansätze, or measurement samplers were executed.
2. **Definitive Answer to RQ3:** Within the scope of QRadAssist, a classical linear classification head is strictly superior to the simulated quantum head in terms of discriminative accuracy (+0.1572 Mean AUROC), training convergence speed (3.0s vs 5.6s), and resilience to class imbalance.
3. **No Modification to Production Assets:** The production DenseNet-121 model remains the primary clinical engine; the simulated quantum head serves as a documented research comparison baseline for the final EDI project submission.
