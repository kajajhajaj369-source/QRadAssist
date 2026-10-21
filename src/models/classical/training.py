"""Training loop for DenseNet-121 multi-label chest X-ray classification.

Design decisions
----------------
- Device selection: CUDA when PyTorch is built with CUDA support and a GPU is
  available; otherwise falls back to CPU.  The installed build is ``2.14.1+cpu``
  which has **no** CUDA support, so all computation happens on CPU.  When a
  CUDA-enabled PyTorch is installed in the future, no code change is needed.
- Checkpoint strategy:
    * An *epoch* checkpoint is always saved after every epoch so that training
      can be inspected or resumed even when AUROC is undefined.
    * The *best* checkpoint (``best_model.pt``) is updated only when the
      average AUROC (over labels with valid values) strictly improves.
    * Both are written atomically via a temp-file-then-rename pattern to avoid
      corrupt files on interrupt.
- AUROC: undefined (None) when fewer than two distinct classes are present in
  the masked validation subset for a label.  Undefined values are *never*
  coerced to 1.0.  Averages exclude undefined values; if *all* values are
  undefined the average is reported as None and the best-checkpoint criterion
  is not satisfied.
- Progress logging: device, dataset sizes, batch size, and epoch count are
  printed at startup.  Batch progress is logged every ``LOG_INTERVAL`` batches.
  Epoch metrics and checkpoint status are logged after every epoch.
- num_workers=0: required on Windows to avoid multiprocessing fork issues.
"""

import time
import random
import tempfile
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from .model import DenseNet121MultiLabel
from .loss import masked_bce_with_logits
from .metrics import compute_auc_ap
from .dataset import CheXpertDataset
from preprocessing.transforms import DeterministicPreprocessor, PreprocessConfig

DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 5
DEFAULT_SEED = 42
LOG_INTERVAL = 50  # log every N batches

import logging

logger = logging.getLogger('train_full')


# Canonical path used by the API loader – do not change without updating app.py.
CHECKPOINT_PATH = Path(__file__).resolve().parents[0] / "best_model.pt"


def _select_device() -> torch.device:
    """Return CUDA device when the installed PyTorch build supports it and a
    compatible GPU is present; otherwise return CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_preprocessor() -> DeterministicPreprocessor:
    # Use default config; it matches settings.yaml defaults
    cfg = PreprocessConfig()
    return DeterministicPreprocessor(cfg)


def _atomic_save(state: dict, path: Path) -> None:
    """Save *state* to *path* atomically using a sibling temp file."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(state, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _validate_checkpoint_path(checkpoint_path: Path, approved_parent: Path) -> None:
    """Raise ValueError if *checkpoint_path* resolves outside *approved_parent*.

    This guards against ``..`` traversal and symlinks that would redirect
    writes to the production checkpoint directory.
    """
    try:
        resolved = checkpoint_path.resolve()
        approved = approved_parent.resolve()
    except Exception as exc:
        raise ValueError(f"Cannot resolve checkpoint path {checkpoint_path!r}: {exc}") from exc
    try:
        resolved.relative_to(approved)
    except ValueError:
        raise ValueError(
            f"Checkpoint path {resolved} is outside the approved directory "
            f"{approved}. Full training must write only to that directory."
        )


def train(
    train_manifest: Path,
    val_manifest: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Path = CHECKPOINT_PATH,
    max_train_samples: int | None = None,
    resume_checkpoint: Path | None = None,
    approved_output_dir: Path | None = None,

) -> tuple[DenseNet121MultiLabel, dict, dict]:
    """Train DenseNet-121 on CheXpert data.

    Returns the trained model and the final validation AUROC/AP dictionaries.
    The *checkpoint_path* receives the best-validation-AUROC model weights.
    An epoch-level resumable checkpoint is written to
    ``<checkpoint_path.parent>/epoch_<N>_ckpt.pt`` after every epoch.

    Parameters
    ----------
    train_manifest:
        Path to the training split CSV manifest.
    val_manifest:
        Path to the validation split CSV manifest.
    batch_size:
        Mini-batch size for training and validation data loaders.
    lr:
        Initial learning rate for the Adam optimiser.
    epochs:
        Number of full training epochs.
    seed:
        RNG seed for reproducibility.
    checkpoint_path:
        Destination for the *best* model state dict.  Epoch checkpoints are
        saved alongside this file.
    max_train_samples:
        Optional cap on the number of training samples (useful for smoke tests).
    approved_output_dir:
        If provided, *checkpoint_path* must resolve inside this directory.
        Pass this from train_full.py to enforce path isolation during full
        training.  Unit tests that use tmp_path directories can omit it.
    """
    # Path isolation guard (only enforced when caller supplies approved_output_dir)
    if approved_output_dir is not None:
        _validate_checkpoint_path(Path(checkpoint_path), Path(approved_output_dir))

    set_seed(seed)
    device = _select_device()

    # ---------------------------------------------------------------------
    # Resume from checkpoint if requested
    # ---------------------------------------------------------------------
    start_epoch = 1
    best_val_auc: float | None = None
    _resume_model_state = None
    _resume_optimizer_state = None
    if resume_checkpoint is not None and resume_checkpoint.is_file():
        chk = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        # Expected keys: epoch, model_state_dict, optimizer_state_dict, best_val_auc (optional)
        start_epoch = int(chk.get('epoch', 0)) + 1
        _resume_model_state = chk.get('model_state_dict')
        _resume_optimizer_state = chk.get('optimizer_state_dict')
        best_val_auc = chk.get('best_val_auc')
        # Log resume info
        logger.info(f"Resuming from checkpoint {resume_checkpoint}: epoch {start_epoch - 1}, best_val_auc={best_val_auc}")
    else:
        logger.info("No resume checkpoint provided; starting from scratch.")

    preprocessor = get_preprocessor()
    train_dataset = CheXpertDataset(train_manifest, preprocessor)
    val_dataset = CheXpertDataset(val_manifest, preprocessor)

    if max_train_samples is not None:
        indices = list(range(min(max_train_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)

    # num_workers=0 is required on Windows (no fork support)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = DenseNet121MultiLabel(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Apply resumed state AFTER model and optimizer are created
    if _resume_model_state is not None:
        model.load_state_dict(_resume_model_state)
        logger.info("Model weights restored from resume checkpoint.")
    if _resume_optimizer_state is not None:
        optimizer.load_state_dict(_resume_optimizer_state)
        logger.info("Optimizer state restored from resume checkpoint.")

    model.train()

    logger.info(f"[Training] Device     : {device}")
    logger.info(f"[Training] Train size : {len(train_dataset)}")
    logger.info(f"[Training] Val size   : {len(val_dataset)}")
    logger.info(f"[Training] Batch size : {batch_size}")
    logger.info(f"[Training] Epochs     : {epochs}")
    logger.info(f"[Training] Batches/ep : {len(train_loader)}")

    # ---------------------------------------------------------------------
    # Training loop starts from start_epoch (may be >1 if resumed)
    # ---------------------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):

        epoch_start = time.time()
        logger.info(f"\n[Epoch {epoch}/{epochs}] Starting training ...")

        # ── Training step ────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = masked_bce_with_logits(logits, targets, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % LOG_INTERVAL == 0 or batch_idx == len(train_loader):
                avg_loss = running_loss / batch_idx
                elapsed = time.time() - epoch_start
                gpu_mem = ""
                if torch.cuda.is_available():
                    used = torch.cuda.memory_reserved(device) // (1024 ** 2)
                    gpu_mem = f"  GPU mem: {used} MiB"
                logger.info(f"[Epoch {epoch}/{epochs}]  batch {batch_idx}/{len(train_loader)}  avg_loss={avg_loss:.4f}  elapsed={elapsed:.0f}s{gpu_mem}")

        # ── Validation step ──────────────────────────────────────────────────
        model.eval()
        all_logits, all_targets, all_masks = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                targets = batch["target"].to(device)
                mask = batch["mask"].to(device)
                logits = model(imgs)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                all_masks.append(mask.cpu())

        val_logits = torch.cat(all_logits)
        val_targets = torch.cat(all_targets)
        val_masks = torch.cat(all_masks)
        val_loss = masked_bce_with_logits(val_logits, val_targets, val_masks).item()
        val_auc, val_ap = compute_auc_ap(val_logits, val_targets, val_masks)

        # Compute average only over *valid* (non-None) AUROC values
        valid_aucs = [v for v in val_auc.values() if v is not None]
        if valid_aucs:
            avg_auc: float | None = float(np.mean(valid_aucs))
        else:
            avg_auc = None  # no valid labels

        undefined_count = len(val_auc) - len(valid_aucs)
        logger.info(f"[Epoch {epoch}/{epochs}] Validation Loss: {val_loss:.4f}  AUROC: {val_auc}  (avg over {len(valid_aucs)} labels: {avg_auc}  undefined: {undefined_count})")

        # ── Epoch checkpoint (always saved) ──────────────────────────────────
        epoch_ckpt_path = checkpoint_path.parent / f"epoch_{epoch}_ckpt.pt"
        epoch_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_auc": val_auc,
            "avg_auc": avg_auc,
            "best_val_auc": best_val_auc,  # carry forward for resume
            "batch_size": batch_size,
            "lr": lr,
        }
        _atomic_save(epoch_state, epoch_ckpt_path)
        logger.info(f"[Epoch {epoch}/{epochs}] Epoch checkpoint saved: {epoch_ckpt_path}")

        # Save best checkpoint and also store best_val_auc in epoch checkpoint for resume
        if avg_auc is not None and (best_val_auc is None or avg_auc > best_val_auc):
            best_val_auc = avg_auc
            # Save only state dict for API compatibility
            _atomic_save(model.state_dict(), checkpoint_path)
            logger.info(f"[Epoch {epoch}/{epochs}] Best model updated (avg_auc={avg_auc:.4f}): {checkpoint_path}")
        elif avg_auc is None:
            logger.info(f"[Epoch {epoch}/{epochs}] AUROC undefined for all labels – best-model checkpoint NOT updated.")
        else:
            logger.info(f"[Epoch {epoch}/{epochs}] avg_auc={avg_auc:.4f} did not improve (best={best_val_auc:.4f}) – best-model checkpoint not updated.")


    # Load best model state dict if checkpoint exists
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    logger.info("Training completed. Model returned.")
    return model, val_auc, val_ap
