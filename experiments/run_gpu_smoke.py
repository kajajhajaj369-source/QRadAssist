"""Controlled GPU smoke training run for QRadAssist Phase 9.

This script executes a strict 1-epoch, 300-sample smoke run using:
- Physical GPU: NVIDIA GeForce GTX 1650 (CUDA)
- Model: DenseNet121MultiLabel
- Actual CheXpertDataset, masked_bce_with_logits, compute_auc_ap
- Isolated output path: experiments/smoke_gpu/
Existing checkpoints and backups are strictly untouched.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.classical.training import train, _select_device
from models.classical.model import DenseNet121MultiLabel, TARGET_LABELS

EXPERIMENT_DIR = ROOT / "experiments" / "smoke_gpu"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_CHECKPOINT = EXPERIMENT_DIR / "smoke_best_model.pt"
EPOCH_CHECKPOINT = EXPERIMENT_DIR / "epoch_1_ckpt.pt"

TRAIN_MANIFEST = ROOT / "data" / "splits" / "train_manifest.csv"
FULL_VAL_MANIFEST = ROOT / "data" / "splits" / "val_manifest.csv"

# Create a 1000-sample validation subset manifest so validation is representative and fast (~7s)
SMOKE_VAL_MANIFEST = EXPERIMENT_DIR / "val_smoke_1000.csv"
val_df = pd.read_csv(FULL_VAL_MANIFEST)
val_df.iloc[:1000].to_csv(SMOKE_VAL_MANIFEST, index=False)

print("=" * 60)
print("QRadAssist Phase 9 — Controlled GPU Smoke Training Run")
print("=" * 60)

device = _select_device()
print(f"Target Device        : {device}")
if device.type == "cuda":
    print(f"GPU Name             : {torch.cuda.get_device_name(0)}")
    print(f"Total GPU VRAM       : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MiB")
    torch.cuda.reset_peak_memory_stats(device)

BATCH_SIZE = 8
EPOCHS = 1
TRAIN_SAMPLES = 300
VAL_SAMPLES = 1000

print(f"Batch Size           : {BATCH_SIZE}")
print(f"Epochs               : {EPOCHS}")
print(f"Train Samples        : {TRAIN_SAMPLES}")
print(f"Val Samples          : {VAL_SAMPLES}")
print(f"Output Checkpoint    : {SMOKE_CHECKPOINT}")
print("=" * 60)

t0 = time.time()

model, val_auc, val_ap = train(
    train_manifest=TRAIN_MANIFEST,
    val_manifest=SMOKE_VAL_MANIFEST,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    max_train_samples=TRAIN_SAMPLES,
    checkpoint_path=SMOKE_CHECKPOINT,
)

t1 = time.time()
elapsed_total = t1 - t0

peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
peak_res = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0

print("\n" + "=" * 60)
print("Smoke Run Completed Successfully")
print("=" * 60)
print(f"Total Elapsed Time   : {elapsed_total:.2f} s")
print(f"Peak VRAM Allocated  : {peak_alloc:.1f} MiB")
print(f"Peak VRAM Reserved   : {peak_res:.1f} MiB")

print("\nValidation AUROC (SMOKE TEST ONLY - NOT CLINICAL PERFORMANCE):")
for label, score in val_auc.items():
    print(f"  {label:<18}: {score}")

print("\nValidation Average Precision (SMOKE TEST ONLY):")
for label, score in val_ap.items():
    print(f"  {label:<18}: {score}")

# --- Verification of generated checkpoints ---
print("\n" + "=" * 60)
print("Verifying Checkpoint Integrity & Resumability")
print("=" * 60)

assert SMOKE_CHECKPOINT.exists(), f"Error: {SMOKE_CHECKPOINT} was not created!"
assert EPOCH_CHECKPOINT.exists(), f"Error: {EPOCH_CHECKPOINT} was not created!"

print(f"Best Checkpoint File : {SMOKE_CHECKPOINT.name} ({SMOKE_CHECKPOINT.stat().st_size} bytes)")
print(f"Epoch Checkpoint File: {EPOCH_CHECKPOINT.name} ({EPOCH_CHECKPOINT.stat().st_size} bytes)")

# 1. Verify loading into fresh model
fresh_model = DenseNet121MultiLabel(pretrained=False).to(device)
fresh_state = torch.load(SMOKE_CHECKPOINT, map_location=device, weights_only=True)
fresh_model.load_state_dict(fresh_state)
fresh_model.eval()

# Check all weights are finite
all_finite = True
for name, p in fresh_model.named_parameters():
    if not torch.isfinite(p).all():
        all_finite = False
        print(f"Non-finite weight detected: {name}")
print(f"All model weights finite: {all_finite}")
assert all_finite

# Forward test with fresh model
dummy = torch.randn(2, 1, 224, 224, device=device)
with torch.no_grad():
    out = fresh_model(dummy)
print(f"Fresh model test pass output shape: {out.shape} (finite: {torch.isfinite(out).all().item()})")
assert out.shape == (2, len(TARGET_LABELS))

# 2. Verify resumable epoch checkpoint
epoch_state = torch.load(EPOCH_CHECKPOINT, map_location=device, weights_only=False)
required_keys = ("epoch", "model_state_dict", "optimizer_state_dict", "val_auc", "avg_auc", "batch_size", "lr")
for k in required_keys:
    assert k in epoch_state, f"Missing key '{k}' in epoch checkpoint!"
print(f"Epoch Checkpoint Keys Valid: {list(epoch_state.keys())}")
print(f"Saved Epoch: {epoch_state['epoch']}, Saved Avg AUROC: {epoch_state['avg_auc']}")

print("\n>>> ALL SMOKE RUN VERIFICATIONS PASSED <<<")
