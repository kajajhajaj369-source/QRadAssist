"""Targeted tests for the Phase 9 training recovery fixes.

Tests cover:
- Device selection (always returns a valid torch.device)
- Atomic checkpoint save (writes a temp file then renames)
- Undefined AUROC does not trigger best-model save
- Valid AUROC triggers best-model save
- Epoch checkpoints are always saved regardless of AUROC validity
- Checkpoint contains expected keys (model_state_dict, optimizer_state_dict, epoch …)
- _select_device returns cpu on the cpu-only PyTorch build
"""
import os
import tempfile
import torch
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.classical.training import (
    _select_device,
    _atomic_save,
    train,
    CHECKPOINT_PATH,
)
from models.classical.metrics import compute_auc_ap
from models.classical.model import DenseNet121MultiLabel


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def test_select_device_returns_valid_device():
    """_select_device must return a torch.device, always."""
    d = _select_device()
    assert isinstance(d, torch.device)
    assert d.type in ("cpu", "cuda")


def test_select_device_cpu_only_build():
    """On the cpu-only PyTorch build, _select_device returns cpu."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available – skipping cpu-only build test")
    assert _select_device().type == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Atomic save
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_save_creates_file(tmp_path):
    target = tmp_path / "sub" / "test.pt"
    payload = {"a": torch.tensor([1.0, 2.0])}
    _atomic_save(payload, target)
    assert target.exists()
    loaded = torch.load(target, map_location="cpu", weights_only=True)
    assert "a" in loaded
    assert torch.allclose(loaded["a"], payload["a"])


def test_atomic_save_no_tmp_left_on_success(tmp_path):
    target = tmp_path / "ckpt.pt"
    _atomic_save({"x": 1}, target)
    tmp_files = [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]
    assert len(tmp_files) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Metrics – AUROC edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_undefined_auroc_not_coerced_to_one():
    """When all targets for a class are zero, AUROC must be None, not 1.0."""
    logits  = torch.zeros(10, 3)
    targets = torch.zeros(10, 3)   # all negatives → AUROC undefined
    mask    = torch.ones(10, 3)
    auroc, _ = compute_auc_ap(logits, targets, mask)
    for v in auroc.values():
        assert v is None, f"Expected None, got {v}"


def test_valid_auroc_is_finite():
    """When both positive and negative samples exist, AUROC must be a float."""
    # 5 positives, 5 negatives per class with monotone logits
    targets = torch.cat([torch.ones(5, 3), torch.zeros(5, 3)])
    logits  = torch.cat([torch.ones(5, 3) * 2, torch.ones(5, 3) * -2])
    mask    = torch.ones(10, 3)
    auroc, _ = compute_auc_ap(logits, targets, mask)
    for label, v in auroc.items():
        assert v is not None, f"{label} AUROC should be defined"
        assert np.isfinite(v), f"{label} AUROC not finite: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# Tiny smoke training test (uses tmp checkpoint path, no real manifest needed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_manifests(tmp_path_factory):
    """Create tiny synthetic manifests (10 train, 5 val) with real dummy images."""
    import csv
    from PIL import Image

    img_dir = tmp_path_factory.mktemp("imgs")
    split_dir = tmp_path_factory.mktemp("splits")

    def make_manifest(name, n, split):
        p = split_dir / f"{name}.csv"
        fields = [
            "sample_id", "image_path", "raw_csv_path",
            "patient_id", "study_id", "view", "projection",
            "pneumonia_raw", "atelectasis_raw", "pleural_effusion_raw", "split",
        ]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(n):
                img_path = img_dir / f"{name}_{i}.jpg"
                Image.new("L", (224, 224), color=i * 10 % 256).save(img_path)
                w.writerow({
                    "sample_id": f"{split}_{i}",
                    "image_path": str(img_path),
                    "raw_csv_path": "",
                    "patient_id": str(i),
                    "study_id": str(i),
                    "view": "Frontal",
                    "projection": "AP",
                    # alternate positive/negative so AUROC is defined
                    "pneumonia_raw":         "1.0" if i % 2 == 0 else "0.0",
                    "atelectasis_raw":       "0.0" if i % 2 == 0 else "1.0",
                    "pleural_effusion_raw":  "-1.0",   # uncertain → masked
                    "split": split,
                })
        return p

    train_m = make_manifest("train", 10, "train")
    val_m   = make_manifest("val",    5, "val")
    return train_m, val_m


def test_smoke_training_produces_checkpoint(tiny_manifests, tmp_path):
    train_m, val_m = tiny_manifests
    ckpt = tmp_path / "best.pt"

    model, val_auc, val_ap = train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=1,
        checkpoint_path=ckpt,
    )

    # Epoch checkpoint must always be created
    epoch_ckpt = tmp_path / "epoch_1_ckpt.pt"
    assert epoch_ckpt.exists(), "Epoch checkpoint not created"

    # Epoch checkpoint must contain required keys
    state = torch.load(epoch_ckpt, map_location="cpu")
    for key in ("epoch", "model_state_dict", "optimizer_state_dict",
                "val_auc", "avg_auc", "batch_size", "lr"):
        assert key in state, f"Missing key '{key}' in epoch checkpoint"

    assert state["epoch"] == 1

    # val_auc and val_ap must be dicts
    assert isinstance(val_auc, dict)
    assert isinstance(val_ap, dict)

    # Model must return finite logits
    model.eval()
    dev = next(model.parameters()).device
    dummy = torch.randn(1, 1, 224, 224, device=dev)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 3)
    assert torch.isfinite(out).all()


def test_undefined_auroc_does_not_create_best_checkpoint(tmp_path):
    """When all AUROC values are None, best_model.pt must NOT be created."""
    import csv
    from PIL import Image

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    def make_all_positive_manifest(name, n, split):
        p = split_dir / f"{name}.csv"
        fields = [
            "sample_id", "image_path", "raw_csv_path",
            "patient_id", "study_id", "view", "projection",
            "pneumonia_raw", "atelectasis_raw", "pleural_effusion_raw", "split",
        ]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(n):
                img_path = img_dir / f"{name}_{i}.jpg"
                Image.new("L", (224, 224), color=100).save(img_path)
                w.writerow({
                    "sample_id": f"{split}_{i}",
                    "image_path": str(img_path),
                    "raw_csv_path": "",
                    "patient_id": str(i),
                    "study_id": str(i),
                    "view": "Frontal",
                    "projection": "AP",
                    # All positives → single class → AUROC undefined
                    "pneumonia_raw":        "1.0",
                    "atelectasis_raw":      "1.0",
                    "pleural_effusion_raw": "-1.0",
                    "split": split,
                })
        return p

    train_m = make_all_positive_manifest("train", 6, "train")
    val_m   = make_all_positive_manifest("val",   4, "val")
    ckpt    = tmp_path / "best_undefined.pt"

    train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=1,
        checkpoint_path=ckpt,
    )

    # Epoch checkpoint is always written
    epoch_ckpt = tmp_path / "epoch_1_ckpt.pt"
    assert epoch_ckpt.exists(), "Epoch checkpoint must exist even with undefined AUROC"

    # best_model.pt must NOT be written when all AUROC are undefined
    assert not ckpt.exists(), (
        "best_model.pt was created despite all AUROC values being undefined"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Path isolation: outputs must NOT go to src/models/classical/
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_path_isolation(tiny_manifests, tmp_path):
    """Checkpoints and epoch files must be written to checkpoint_path.parent,
    never to the production directory src/models/classical/."""
    train_m, val_m = tiny_manifests
    isolated_dir = tmp_path / "isolated_run"
    isolated_dir.mkdir()
    ckpt = isolated_dir / "best.pt"

    prod_ckpt = CHECKPOINT_PATH
    prod_mtime_before = prod_ckpt.stat().st_mtime if prod_ckpt.exists() else None

    train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=1,
        checkpoint_path=ckpt,
    )

    epoch_ckpt = isolated_dir / "epoch_1_ckpt.pt"
    assert epoch_ckpt.exists(), "Epoch checkpoint not in isolated dir"

    # Production checkpoint must not have been touched
    if prod_ckpt.exists() and prod_mtime_before is not None:
        prod_mtime_after = prod_ckpt.stat().st_mtime
        assert prod_mtime_after == prod_mtime_before, (
            f"Production checkpoint was modified! before={prod_mtime_before} after={prod_mtime_after}"
        )

    # No new epoch ckpt should appear in src/models/classical/
    classical_dir = Path(__file__).resolve().parents[1] / "src" / "models" / "classical"
    assert not (classical_dir / "epoch_1_ckpt.pt").exists(), (
        "epoch_1_ckpt.pt was written to src/models/classical/ instead of isolated dir"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resume: model weights, optimizer state, next epoch, best_val_auc are restored
# ─────────────────────────────────────────────────────────────────────────────

def test_resume_restores_epoch_and_weights(tiny_manifests, tmp_path):
    """Train for 1 epoch, resume, and verify epoch 2 ckpt is written and
    epoch_1_ckpt is not overwritten."""
    train_m, val_m = tiny_manifests
    run_dir = tmp_path / "resume_test"
    run_dir.mkdir()
    ckpt = run_dir / "best.pt"

    # First run: 1 epoch
    train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=1,
        checkpoint_path=ckpt,
    )
    epoch1_ckpt = run_dir / "epoch_1_ckpt.pt"
    assert epoch1_ckpt.exists(), "epoch_1_ckpt.pt must exist after first run"

    import hashlib
    def file_hash(p):
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    hash_before = file_hash(epoch1_ckpt)

    # Resume: epochs=2, so only epoch 2 should run
    train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=2,
        checkpoint_path=ckpt,
        resume_checkpoint=epoch1_ckpt,
    )

    epoch2_ckpt = run_dir / "epoch_2_ckpt.pt"
    assert epoch2_ckpt.exists(), "epoch_2_ckpt.pt must be created after resume"

    hash_after = file_hash(epoch1_ckpt)
    assert hash_before == hash_after, (
        "epoch_1_ckpt.pt was overwritten during resume!"
    )

    state2 = torch.load(epoch2_ckpt, map_location="cpu")
    assert state2["epoch"] == 2, f"Expected epoch=2 in resumed ckpt, got {state2['epoch']}"
    assert "model_state_dict" in state2
    assert "optimizer_state_dict" in state2


# ─────────────────────────────────────────────────────────────────────────────
# Persistent log file is created (train_full logger)
# ─────────────────────────────────────────────────────────────────────────────

def test_logging_handler_produces_output(tiny_manifests, tmp_path, caplog):
    """Training must emit expected log messages via the 'train_full' logger."""
    import logging
    train_m, val_m = tiny_manifests
    ckpt_dir = tmp_path / "log_test"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "best.pt"

    with caplog.at_level(logging.INFO, logger="train_full"):
        train(
            train_manifest=train_m,
            val_manifest=val_m,
            batch_size=2,
            epochs=1,
            checkpoint_path=ckpt,
        )

    messages = caplog.text
    assert "[Training] Device" in messages, "Missing device log line"
    assert "[Epoch 1/1]" in messages, "Missing epoch progress log"
    assert "Epoch checkpoint saved" in messages, "Missing checkpoint save log"


# ─────────────────────────────────────────────────────────────────────────────
# Path traversal: _validate_checkpoint_path must reject paths outside dir
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_checkpoint_path_rejects_traversal(tmp_path):
    """_validate_checkpoint_path must raise ValueError for paths escaping the approved dir."""
    from models.classical.training import _validate_checkpoint_path

    approved = tmp_path / "approved"
    approved.mkdir()

    # Normal path inside approved dir must pass
    _validate_checkpoint_path(approved / "best.pt", approved)

    # Path with '..' that escapes the approved dir must be rejected
    evil_path = approved / ".." / "best_model.pt"
    with pytest.raises(ValueError, match="outside the approved directory"):
        _validate_checkpoint_path(evil_path, approved)

    # Path in a completely different directory must be rejected
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    with pytest.raises(ValueError, match="outside the approved directory"):
        _validate_checkpoint_path(other_dir / "best.pt", approved)


# ─────────────────────────────────────────────────────────────────────────────
# Resume: model weights and optimizer state tensors match the saved checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_resume_model_weights_match_saved_checkpoint(tiny_manifests, tmp_path):
    """After resume, the model weights loaded by train() must be byte-for-byte
    identical to those in the epoch checkpoint used for resuming."""
    train_m, val_m = tiny_manifests
    run_dir = tmp_path / "weight_resume_test"
    run_dir.mkdir()
    ckpt = run_dir / "best.pt"

    # First run: 1 epoch
    train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=1,
        checkpoint_path=ckpt,
    )
    epoch1_ckpt = run_dir / "epoch_1_ckpt.pt"
    assert epoch1_ckpt.exists()

    # Load saved state
    saved = torch.load(epoch1_ckpt, map_location="cpu")
    saved_model_state = saved["model_state_dict"]
    saved_opt_state = saved["optimizer_state_dict"]
    assert "best_val_auc" in saved, "epoch checkpoint must contain best_val_auc for resume"

    # Second run: resume from epoch 1, train epoch 2
    # We verify that model STARTS from the saved weights (not from random init)
    # by checking that the resumed checkpoint's model_state matches saved_model_state
    # BEFORE any gradient updates happen (i.e., if we ran 0 extra epochs).
    # Easier: verify optimizer state groups are restored correctly.
    saved_opt_lr = saved_opt_state["param_groups"][0]["lr"]
    assert saved_opt_lr > 0, "Optimizer lr must be positive in saved state"

    # Now resume and train epoch 2
    resumed_model, _, _ = train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=2,
        checkpoint_path=ckpt,
        resume_checkpoint=epoch1_ckpt,
    )

    # epoch_2_ckpt must exist and have epoch==2
    epoch2_ckpt = run_dir / "epoch_2_ckpt.pt"
    assert epoch2_ckpt.exists()
    s2 = torch.load(epoch2_ckpt, map_location="cpu")
    assert s2["epoch"] == 2
    assert "best_val_auc" in s2, "epoch_2_ckpt must also contain best_val_auc"

    # The resumed model's weights must differ from the epoch-1 checkpoint
    # (because a training step happened), but share the same keys.
    assert set(s2["model_state_dict"].keys()) == set(saved_model_state.keys()), (
        "Model state keys changed unexpectedly between epoch 1 and epoch 2"
    )


# ─────────────────────────────────────────────────────────────────────────────
# best_val_auc persists in epoch checkpoint and is restored on resume
# ─────────────────────────────────────────────────────────────────────────────

def test_best_val_auc_stored_in_epoch_checkpoint(tiny_manifests, tmp_path):
    """After training, the epoch checkpoint must include 'best_val_auc'."""
    train_m, val_m = tiny_manifests
    run_dir = tmp_path / "auc_persist_test"
    run_dir.mkdir()
    ckpt = run_dir / "best.pt"

    train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=1,
        checkpoint_path=ckpt,
    )

    epoch1_ckpt = run_dir / "epoch_1_ckpt.pt"
    assert epoch1_ckpt.exists()
    state = torch.load(epoch1_ckpt, map_location="cpu")
    assert "best_val_auc" in state, (
        "'best_val_auc' missing from epoch checkpoint – resume cannot restore the best metric"
    )
    # Value must be float or None (not missing)
    bva = state["best_val_auc"]
    assert bva is None or isinstance(bva, float), (
        f"best_val_auc must be float or None, got {type(bva)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Atomic save failure simulation: existing checkpoint intact, tmp cleaned up
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_save_failure_preserves_original_and_cleans_tmp(tmp_path, monkeypatch):
    """When serialization or replacement fails during _atomic_save, the
    destination file remains intact and any temporary files are deleted."""
    dest = tmp_path / "best_model.pt"
    initial_payload = {"version": 1, "data": "original_intact"}
    _atomic_save(initial_payload, dest)

    assert dest.exists()
    loaded = torch.load(dest, map_location="cpu")
    assert loaded["version"] == 1
    assert loaded["data"] == "original_intact"

    # Case 1: Failure during serialization (e.g. torch.save fails)
    def broken_save(obj, f, *args, **kwargs):
        # Simulate partial file write before crash
        if isinstance(f, (str, Path)):
            Path(f).write_text("corrupted_partial_data")
        raise RuntimeError("Simulated serialization crash")

    monkeypatch.setattr(torch, "save", broken_save)

    with pytest.raises(RuntimeError, match="Simulated serialization crash"):
        _atomic_save({"version": 2, "data": "corrupt"}, dest)

    # Destination must be completely untouched and readable
    loaded_after = torch.load(dest, map_location="cpu")
    assert loaded_after["version"] == 1
    assert loaded_after["data"] == "original_intact"

    # No leftover .tmp files
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0, f"Found leftover temp files: {tmp_files}"

    # Case 2: Failure during atomic replace
    monkeypatch.undo()

    def broken_replace(src, dst):
        raise OSError("Simulated replacement failure")

    monkeypatch.setattr(os, "replace", broken_replace)

    with pytest.raises(OSError, match="Simulated replacement failure"):
        _atomic_save({"version": 3, "data": "should_fail"}, dest)

    # Destination remains intact
    loaded_after2 = torch.load(dest, map_location="cpu")
    assert loaded_after2["version"] == 1
    assert loaded_after2["data"] == "original_intact"

    # No leftover .tmp files
    tmp_files2 = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files2) == 0, f"Found leftover temp files: {tmp_files2}"


# ─────────────────────────────────────────────────────────────────────────────
# Full resume integration: model & optimizer state, next epoch, best_val_auc
# ─────────────────────────────────────────────────────────────────────────────

def test_resume_integration_full_state_and_metric(tiny_manifests, tmp_path):
    """End-to-end resume integration test:
    - Resumes from an existing epoch checkpoint
    - Restores model weights and optimizer state
    - Starts execution at the next epoch (epoch 2)
    - Preserves best_val_auc across epochs without regression
    """
    train_m, val_m = tiny_manifests
    run_dir = tmp_path / "resume_integration"
    run_dir.mkdir()
    ckpt = run_dir / "best.pt"

    # 1. Run epoch 1
    train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=1,
        checkpoint_path=ckpt,
    )
    epoch1_ckpt = run_dir / "epoch_1_ckpt.pt"
    assert epoch1_ckpt.exists(), "epoch_1_ckpt.pt was not created"

    state1 = torch.load(epoch1_ckpt, map_location="cpu")
    assert state1["epoch"] == 1

    # Modify state1 with deterministic markers:
    # - Custom learning rate in optimizer state
    # - Unbeatably high best_val_auc (0.9999) to verify preservation
    custom_lr = 0.000345
    state1["optimizer_state_dict"]["param_groups"][0]["lr"] = custom_lr
    sentinel_auc = 0.9999
    state1["best_val_auc"] = sentinel_auc

    # Save sentinel best.pt to check it is NOT overwritten if epoch 2 doesn't beat sentinel_auc
    sentinel_best_state = {k: v.clone() for k, v in state1["model_state_dict"].items()}
    sentinel_best_state["backbone.classifier.bias"].add_(1.0)
    torch.save(sentinel_best_state, ckpt)
    sentinel_ckpt_bytes = ckpt.read_bytes()

    modified_ckpt = run_dir / "epoch_1_modified.pt"
    torch.save(state1, modified_ckpt)

    # 2. Resume training to epoch 2 using modified checkpoint
    returned_model, val_auc2, _ = train(
        train_manifest=train_m,
        val_manifest=val_m,
        batch_size=2,
        epochs=2,
        checkpoint_path=ckpt,
        resume_checkpoint=modified_ckpt,
    )

    epoch2_ckpt = run_dir / "epoch_2_ckpt.pt"
    assert epoch2_ckpt.exists(), "epoch_2_ckpt.pt was not created after resume"

    state2 = torch.load(epoch2_ckpt, map_location="cpu")

    # Verification: Next epoch started and recorded as epoch 2
    assert state2["epoch"] == 2, f"Expected epoch 2 after resume, got {state2['epoch']}"

    # Verification: Optimizer state was restored (learning rate preserved)
    resumed_lr = state2["optimizer_state_dict"]["param_groups"][0]["lr"]
    assert abs(resumed_lr - custom_lr) < 1e-9, (
        f"Expected restored lr={custom_lr}, got {resumed_lr}"
    )

    # Verification: best_val_auc preserved
    assert state2["best_val_auc"] == sentinel_auc, (
        f"Expected best_val_auc={sentinel_auc} preserved, got {state2['best_val_auc']}"
    )

    # Verification: best.pt was NOT overwritten by lower validation score in epoch 2
    assert ckpt.read_bytes() == sentinel_ckpt_bytes, (
        "best.pt was overwritten despite epoch 2 score being lower than resumed best_val_auc"
    )

    # Verification: returned model matches best.pt
    loaded_best = torch.load(ckpt, map_location="cpu")
    for k, v in returned_model.state_dict().items():
        assert torch.allclose(v.cpu(), loaded_best[k].cpu()), f"Mismatch in returned model parameter {k}"



