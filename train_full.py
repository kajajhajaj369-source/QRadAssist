"""Full-dataset training script for QRadAssist Phase 9.

Usage
-----
  python train_full.py [--smoke] [--resume-checkpoint <path>]

Flags
-----
  --smoke               Run a quick smoke test with 300 samples / 1 epoch instead of the full training run.
  --resume-checkpoint   Path to an epoch checkpoint file to resume training from.

The script now writes all checkpoints and logs to the isolated `experiments/full_gpu/` directory,
never touching the production checkpoint `src/models/classical/best_model.pt` or any backup files.
"""

import sys
import argparse
from pathlib import Path
import logging

# Ensure src/ is on PYTHONPATH so that imports from src/* work regardless of how/where the script is invoked.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from models.classical.training import train

TRAIN_MANIFEST = ROOT / "data" / "splits" / "train_manifest.csv"
VAL_MANIFEST   = ROOT / "data" / "splits" / "val_manifest.csv"

FULL_EPOCHS      = 5
FULL_BATCH_SIZE  = 8   # safe for 4 GB VRAM when CUDA is available

SMOKE_EPOCHS     = 1
SMOKE_SAMPLES    = 300
SMOKE_BATCH_SIZE = 4

# Directory where full-GPU training artifacts will be stored.
FULL_OUTPUT_DIR = ROOT / "experiments" / "full_gpu"

# Paths used by this script – they are **isolated** from the production checkpoint.
FULL_BEST_CKPT = FULL_OUTPUT_DIR / "best_full_model.pt"
FULL_LOG_FILE  = FULL_OUTPUT_DIR / "training.log"

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("train_full")
    logger.setLevel(logging.INFO)
    # Avoid adding duplicate handlers if this function is called multiple times.
    if not logger.handlers:
        stream = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(message)s")
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(FULL_LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QRadAssist DenseNet-121 (full dataset)")
    p.add_argument("--smoke", action="store_true", help="Quick smoke test (300 samples, 1 epoch)")
    p.add_argument("--resume-checkpoint", type=Path, help="Path to an epoch checkpoint to resume training from")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    # Create output directory here (inside main), not at module import time.
    FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger()

    if args.smoke:
        logger.info("=== SMOKE TEST (300 samples, 1 epoch) ===")
        model, val_auc, val_ap = train(
            train_manifest=TRAIN_MANIFEST,
            val_manifest=VAL_MANIFEST,
            batch_size=SMOKE_BATCH_SIZE,
            epochs=SMOKE_EPOCHS,
            max_train_samples=SMOKE_SAMPLES,
            checkpoint_path=FULL_BEST_CKPT,
            resume_checkpoint=args.resume_checkpoint,
            approved_output_dir=FULL_OUTPUT_DIR,
        )
    else:
        logger.info("=== FULL TRAINING (all samples, 5 epochs) ===")
        model, val_auc, val_ap = train(
            train_manifest=TRAIN_MANIFEST,
            val_manifest=VAL_MANIFEST,
            batch_size=FULL_BATCH_SIZE,
            epochs=FULL_EPOCHS,
            checkpoint_path=FULL_BEST_CKPT,
            resume_checkpoint=args.resume_checkpoint,
            approved_output_dir=FULL_OUTPUT_DIR,
        )

    logger.info("\n=== Final Validation AUROC ===")
    for label, auc in val_auc.items():
        logger.info(f"  {label}: {auc}")

    logger.info("\n=== Final Validation Average Precision ===")
    for label, ap in val_ap.items():
        logger.info(f"  {label}: {ap}")

    valid_aucs = [v for v in val_auc.values() if v is not None]
    import numpy as np
    avg = float(np.mean(valid_aucs)) if valid_aucs else None
    logger.info(f"\nMean AUROC ({len(valid_aucs)}/{len(val_auc)} labels): {avg}")

if __name__ == "__main__":
    main()
