"""Targeted GPU development training run for QRadAssist Phase 9.

Configuration:
- 5,000 deterministic training samples from data/splits/train_manifest.csv
- 1,500 separate validation samples from data/splits/val_manifest.csv (no patient overlap)
- 3 full epochs, batch size 8, 224x224 input resolution
- Device: CUDA (NVIDIA GeForce GTX 1650, 4 GB VRAM)
- Output: experiments/dev_gpu/best_dev_model.pt, epoch_{1,2,3}_ckpt.pt
- Telemetry: train loss, val loss, per-label AUROC/AP, peak VRAM, epoch elapsed time
- All existing checkpoints, backups, and smoke artifacts are strictly preserved.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.classical.training import train, _select_device
from models.classical.model import DenseNet121MultiLabel, TARGET_LABELS

DEV_DIR = ROOT / "experiments" / "dev_gpu"
DEV_DIR.mkdir(parents=True, exist_ok=True)

BEST_DEV_CHECKPOINT = DEV_DIR / "best_dev_model.pt"

TRAIN_MANIFEST = ROOT / "data" / "splits" / "train_manifest.csv"
FULL_VAL_MANIFEST = ROOT / "data" / "splits" / "val_manifest.csv"

# Create a separate, deterministic 1500-sample validation subset manifest
DEV_VAL_MANIFEST = DEV_DIR / "val_dev_1500.csv"
val_df = pd.read_csv(FULL_VAL_MANIFEST)
val_df.iloc[:1500].to_csv(DEV_VAL_MANIFEST, index=False)

print("=" * 65)
print("QRadAssist Phase 9 — Targeted GPU Development Training (3 Epochs)")
print("=" * 65)

device = _select_device()
print(f"Target Device        : {device}")
if device.type == "cuda":
    print(f"GPU Name             : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM           : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MiB")
    torch.cuda.reset_peak_memory_stats(device)

BATCH_SIZE = 8
EPOCHS = 3
TRAIN_SAMPLES = 5000
VAL_SAMPLES = 1500

print(f"Batch Size           : {BATCH_SIZE}")
print(f"Epochs               : {EPOCHS}")
print(f"Training Samples     : {TRAIN_SAMPLES}")
print(f"Validation Samples   : {VAL_SAMPLES}")
print(f"Checkpoint Target    : {BEST_DEV_CHECKPOINT}")
print("=" * 65)

t_start = time.time()

model, val_auc, val_ap = train(
    train_manifest=TRAIN_MANIFEST,
    val_manifest=DEV_VAL_MANIFEST,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    max_train_samples=TRAIN_SAMPLES,
    checkpoint_path=BEST_DEV_CHECKPOINT,
)

t_end = time.time()
total_time = t_end - t_start

peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
peak_res = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0

print("\n" + "=" * 65)
print("Development Training Complete")
print("=" * 65)
print(f"Total Elapsed Time   : {total_time:.2f} s ({total_time/60:.2f} min)")
print(f"Peak VRAM Allocated  : {peak_alloc:.1f} MiB")
print(f"Peak VRAM Reserved   : {peak_res:.1f} MiB")

print("\nFinal Validation AUROC (Development Experiment - NOT Clinical Validation):")
for label, score in val_auc.items():
    print(f"  {label:<18}: {score}")

valid_aucs = [v for v in val_auc.values() if v is not None]
mean_auc = float(np.mean(valid_aucs)) if valid_aucs else None
print(f"  Mean AUROC        : {mean_auc}")

print("\nFinal Validation Average Precision (Development Experiment):")
for label, score in val_ap.items():
    print(f"  {label:<18}: {score}")

# --- Verification of Checkpoints ---
print("\n" + "=" * 65)
print("Verifying Development Checkpoints and Resumability")
print("=" * 65)

assert BEST_DEV_CHECKPOINT.exists(), f"Missing {BEST_DEV_CHECKPOINT}"
for ep in range(1, EPOCHS + 1):
    ep_path = DEV_DIR / f"epoch_{ep}_ckpt.pt"
    assert ep_path.exists(), f"Missing epoch checkpoint {ep_path}"
    st = torch.load(ep_path, map_location=device, weights_only=False)
    for k in ("epoch", "model_state_dict", "optimizer_state_dict", "val_loss", "val_auc", "avg_auc", "batch_size", "lr"):
        assert k in st, f"Missing key '{k}' in {ep_path.name}"
    print(f"Verified {ep_path.name}: epoch={st['epoch']}, val_loss={st['val_loss']:.4f}, avg_auc={st['avg_auc']}")

# Verify loading best checkpoint into fresh model
fresh_model = DenseNet121MultiLabel(pretrained=False).to(device)
fresh_state = torch.load(BEST_DEV_CHECKPOINT, map_location=device, weights_only=True)
fresh_model.load_state_dict(fresh_state)
fresh_model.eval()

all_finite = True
for name, p in fresh_model.named_parameters():
    if not torch.isfinite(p).all():
        all_finite = False
        print(f"Non-finite weight detected: {name}")
print(f"All model weights finite: {all_finite}")
assert all_finite

dummy = torch.randn(2, 1, 224, 224, device=device)
with torch.no_grad():
    out = fresh_model(dummy)
print(f"Fresh model evaluation pass output shape: {out.shape} (finite: {torch.isfinite(out).all().item()})")
assert out.shape == (2, len(TARGET_LABELS))

print("\n>>> ALL DEVELOPMENT TRAINING CHECKS PASSED <<<")
