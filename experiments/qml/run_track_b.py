"""Track B Comparative Experiment Runner: Classical vs. Simulated Quantum Head.

Compares:
1. ClassicalLinearHead (1024 -> 3)
2. SimulatedQuantumHead (1024 -> 2 angles -> SimpleQuantumLayer -> 3)

Both operate on identical 1024-dimensional feature embeddings extracted from
the frozen DenseNet-121 model (experiments/full_gpu/best_full_model.pt).
"""

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set up project imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.qml.feature_extractor import (
    EXPECTED_FULL_MODEL_SHA256,
    compute_sha256,
    get_or_extract_features,
)
from experiments.qml.qml_heads import ClassicalLinearHead, SimulatedQuantumHead
from src.models.classical.loss import masked_bce_with_logits
from src.models.classical.metrics import compute_auc_ap
from src.preprocessing.labels import TARGET_LABELS

# Paths and Configuration
QML_DIR = ROOT / "experiments" / "qml"
CACHE_DIR = QML_DIR / "cache"

FULL_MODEL_CHECKPOINT = ROOT / "experiments" / "full_gpu" / "best_full_model.pt"
PROD_MODEL_CHECKPOINT = ROOT / "src" / "models" / "classical" / "best_model.pt"
EXPECTED_PROD_SHA256 = "392b65c9130717bfb19053f5e638ef31bb9fcddc40e169399dbc89cb15b9d25a"

TRAIN_MANIFEST = ROOT / "data" / "splits" / "train_manifest.csv"
VAL_MANIFEST = ROOT / "experiments" / "dev_gpu" / "val_dev_1500.csv"

TRAIN_SAMPLES = 5000
VAL_SAMPLES = 1500

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FIXED_SEED = 42


def set_seed(seed: int = FIXED_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_head(
    head_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    val_masks: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    checkpoint_path: Path = None,
) -> Tuple[Dict, Dict, float]:
    """Train a classification head on cached features and select the best checkpoint."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_mean_auc = -1.0
    best_metrics = {}
    best_epoch = 0

    print(f"\n--- Training {head_name} ({epochs} epochs, lr={lr}) ---")
    t0 = time.time()

    val_features_dev = val_features.to(device)
    val_targets_dev = val_targets.to(device)
    val_masks_dev = val_masks.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for feats, targets, masks in train_loader:
            feats = feats.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = masked_bce_with_logits(logits, targets, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_features_dev)
            val_loss = masked_bce_with_logits(val_logits, val_targets_dev, val_masks_dev).item()
            val_auc, val_ap = compute_auc_ap(val_logits.cpu(), val_targets, val_masks)

        valid_aucs = [v for v in val_auc.values() if v is not None]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

        valid_aps = [v for v in val_ap.values() if v is not None]
        mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "mean_auc": mean_auc,
                "val_ap": val_ap,
                "mean_ap": mean_ap,
            }
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_mean_auc": best_mean_auc,
                        "metrics": best_metrics,
                    },
                    checkpoint_path,
                )

        if epoch % 3 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Mean AUC: {mean_auc:.4f} | Mean AP: {mean_ap:.4f}"
            )

    elapsed = time.time() - t0
    print(
        f"Completed {head_name} in {elapsed:.2f}s | "
        f"Best Epoch: {best_epoch} | Best Mean AUC: {best_mean_auc:.4f}"
    )
    return best_metrics, model, elapsed


def main():
    print("=" * 70)
    print("QRadAssist Track B: Classical vs. Simulated Quantum Head Experiment")
    print("=" * 70)

    # 1. Pre-experiment Checkpoint Integrity Verification
    pre_full_hash = compute_sha256(FULL_MODEL_CHECKPOINT)
    pre_prod_hash = compute_sha256(PROD_MODEL_CHECKPOINT)
    print(f"Pre-check Full Model SHA-256 : {pre_full_hash}")
    print(f"Pre-check Prod Model SHA-256 : {pre_prod_hash}")

    if pre_full_hash != EXPECTED_FULL_MODEL_SHA256:
        raise RuntimeError(f"Full model checkpoint mismatch! Aborting.")
    if pre_prod_hash != EXPECTED_PROD_SHA256:
        raise RuntimeError(f"Production checkpoint mismatch! Aborting.")
    print("Pre-flight integrity: PASS.")

    # 2. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target compute device        : {device}")
    if device.type == "cuda":
        print(f"GPU Model                    : {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM               : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MiB")
        torch.cuda.reset_peak_memory_stats(device)

    # 3. Feature Extraction / Caching
    train_cache = CACHE_DIR / f"features_train_{TRAIN_SAMPLES}.pt"
    val_cache = CACHE_DIR / f"features_val_{VAL_SAMPLES}.pt"

    print("\n--- Feature Extraction & Caching ---")
    train_data = get_or_extract_features(
        manifest_path=TRAIN_MANIFEST,
        checkpoint_path=FULL_MODEL_CHECKPOINT,
        cache_path=train_cache,
        max_samples=TRAIN_SAMPLES,
        device=device,
    )
    val_data = get_or_extract_features(
        manifest_path=VAL_MANIFEST,
        checkpoint_path=FULL_MODEL_CHECKPOINT,
        cache_path=val_cache,
        max_samples=VAL_SAMPLES,
        device=device,
    )

    train_feats = train_data["features"]
    train_targets = train_data["targets"]
    train_masks = train_data["masks"]

    val_feats = val_data["features"]
    val_targets = val_data["targets"]
    val_masks = val_data["masks"]

    print(f"\nTrain Features Shape: {train_feats.shape}, Targets: {train_targets.shape}")
    print(f"Val Features Shape  : {val_feats.shape}, Targets: {val_targets.shape}")

    # Report sample counts and label distributions in validation set
    val_counts = {}
    for i, disease in enumerate(TARGET_LABELS):
        mask_i = val_masks[:, i].bool()
        t_i = val_targets[mask_i, i]
        pos = int((t_i == 1.0).sum())
        neg = int((t_i == 0.0).sum())
        masked = int((~mask_i).sum())
        val_counts[disease] = {
            "total": len(val_targets),
            "unmasked": int(mask_i.sum()),
            "positive": pos,
            "negative": neg,
            "masked_uncertain": masked,
        }
        print(f"  Val [{disease}]: {pos} pos, {neg} neg, {masked} masked")

    train_dataset = TensorDataset(train_feats, train_targets, train_masks)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Train Model 1: Classical Linear Head
    set_seed(FIXED_SEED)
    classical_head = ClassicalLinearHead(in_features=1024, num_classes=3)
    classical_ckpt_path = QML_DIR / "best_classical_head.pt"

    classical_metrics, _, classical_time = train_head(
        head_name="Classical Linear Head",
        model=classical_head,
        train_loader=train_loader,
        val_features=val_feats,
        val_targets=val_targets,
        val_masks=val_masks,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        checkpoint_path=classical_ckpt_path,
    )

    # 5. Train Model 2: Simulated Quantum Head
    set_seed(FIXED_SEED)
    quantum_head = SimulatedQuantumHead(in_features=1024, n_qubits=2, num_classes=3)
    quantum_ckpt_path = QML_DIR / "best_quantum_sim_head.pt"

    quantum_metrics, _, quantum_time = train_head(
        head_name="Simulated Quantum Head (2 Qubits)",
        model=quantum_head,
        train_loader=train_loader,
        val_features=val_feats,
        val_targets=val_targets,
        val_masks=val_masks,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        checkpoint_path=quantum_ckpt_path,
    )

    # 6. Peak Memory Usage
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )

    # 7. Post-experiment Integrity Verification
    post_full_hash = compute_sha256(FULL_MODEL_CHECKPOINT)
    post_prod_hash = compute_sha256(PROD_MODEL_CHECKPOINT)
    print("\n--- Checkpoint Integrity Verification ---")
    print(f"Post Full Model SHA-256 : {post_full_hash}")
    print(f"Post Prod Model SHA-256 : {post_prod_hash}")
    assert post_full_hash == EXPECTED_FULL_MODEL_SHA256, "Full model was altered!"
    assert post_prod_hash == EXPECTED_PROD_SHA256, "Production model was altered!"
    print("Post-flight integrity: PASS. Both checkpoints untouched.")

    # 8. Save Metrics JSON
    results = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": "Track B: Feature-Based Classical vs. Simulated Quantum Head",
        "feature_extractor": {
            "checkpoint": str(FULL_MODEL_CHECKPOINT.relative_to(ROOT)),
            "sha256": post_full_hash,
            "feature_dim": 1024,
            "status": "frozen",
        },
        "datasets": {
            "train_manifest": str(TRAIN_MANIFEST.relative_to(ROOT)),
            "train_samples": TRAIN_SAMPLES,
            "val_manifest": str(VAL_MANIFEST.relative_to(ROOT)),
            "val_samples": VAL_SAMPLES,
            "val_counts": val_counts,
        },
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": FIXED_SEED,
            "uncertainty_policy": "u_ignore",
        },
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
            "peak_vram_mib": peak_vram,
        },
        "results": {
            "classical_linear_head": {
                "best_epoch": classical_metrics["epoch"],
                "runtime_seconds": classical_time,
                "train_loss": classical_metrics["train_loss"],
                "val_loss": classical_metrics["val_loss"],
                "val_auc": classical_metrics["val_auc"],
                "mean_auc": classical_metrics["mean_auc"],
                "val_ap": classical_metrics["val_ap"],
                "mean_ap": classical_metrics["mean_ap"],
            },
            "simulated_quantum_head": {
                "best_epoch": quantum_metrics["epoch"],
                "runtime_seconds": quantum_time,
                "train_loss": quantum_metrics["train_loss"],
                "val_loss": quantum_metrics["val_loss"],
                "val_auc": quantum_metrics["val_auc"],
                "mean_auc": quantum_metrics["mean_auc"],
                "val_ap": quantum_metrics["val_ap"],
                "mean_ap": quantum_metrics["mean_ap"],
            },
            "delta_quantum_minus_classical": {
                "mean_auc_diff": quantum_metrics["mean_auc"] - classical_metrics["mean_auc"],
                "mean_ap_diff": quantum_metrics["mean_ap"] - classical_metrics["mean_ap"],
                "per_class_auc_diff": {
                    d: quantum_metrics["val_auc"][d] - classical_metrics["val_auc"][d]
                    for d in TARGET_LABELS
                },
            },
        },
    }

    metrics_file = QML_DIR / "track_b_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}.")

    # 9. Comparison Summary Table
    print("\n" + "=" * 70)
    print("TRACK B EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<18} | {'Classical Head AUROC':<20} | {'Quantum Sim Head AUROC':<22} | {'Diff':<10}")
    print("-" * 75)
    for d in TARGET_LABELS:
        c_auc = classical_metrics["val_auc"][d]
        q_auc = quantum_metrics["val_auc"][d]
        diff = q_auc - c_auc
        print(f"{d:<18} | {c_auc:<20.4f} | {q_auc:<22.4f} | {diff:+7.4f}")
    print("-" * 75)
    print(
        f"{'MACRO MEAN AUROC':<18} | {classical_metrics['mean_auc']:<20.4f} | "
        f"{quantum_metrics['mean_auc']:<22.4f} | {quantum_metrics['mean_auc'] - classical_metrics['mean_auc']:+7.4f}"
    )
    print(
        f"{'MACRO MEAN AP':<18} | {classical_metrics['mean_ap']:<20.4f} | "
        f"{quantum_metrics['mean_ap']:<22.4f} | {quantum_metrics['mean_ap'] - classical_metrics['mean_ap']:+7.4f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
