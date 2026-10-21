import torch
import pytest
from pathlib import Path

from src.models.classical.model import DenseNet121MultiLabel
from src.models.classical.xai import gradcam
from preprocessing.labels import TARGET_LABELS


def test_gradcam_output_shape_and_range():
    model = DenseNet121MultiLabel(pretrained=False)
    # Create a dummy grayscale image tensor (1, 1, 224, 224)
    dummy_input = torch.randn(1, 1, 224, 224)
    # Choose first target class index
    target_idx = 0
    heatmap = gradcam(model, dummy_input, target_idx)
    # Heatmap should be 2D (224, 224)
    assert heatmap.shape == (224, 224)
    # Values should be finite and in [0, 1]
    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_gradcam_invalid_batch_size():
    model = DenseNet121MultiLabel(pretrained=False)
    bad_input = torch.randn(2, 1, 224, 224)  # batch size 2
    with pytest.raises(ValueError, match=r"input_tensor must be shape \(1, C, H, W\)"):
        gradcam(model, bad_input, 0)


def test_gradcam_invalid_target_index():
    model = DenseNet121MultiLabel(pretrained=False)
    dummy_input = torch.randn(1, 1, 224, 224)
    invalid_idx = len(TARGET_LABELS)  # out of range
    with pytest.raises(ValueError, match="out of range"):
        gradcam(model, dummy_input, invalid_idx)


def test_gradcam_class_differentiation():
    """Verify that different pathology target classes yield distinct heatmaps on the same input."""
    torch.manual_seed(123)
    model = DenseNet121MultiLabel(pretrained=False)
    dummy_input = torch.randn(1, 1, 224, 224)

    hm_pneu = gradcam(model, dummy_input, 0)
    hm_atel = gradcam(model, dummy_input, 1)
    hm_effu = gradcam(model, dummy_input, 2)

    assert not torch.allclose(hm_pneu, hm_atel), "Pneumonia and Atelectasis heatmaps should differ"
    assert not torch.allclose(hm_atel, hm_effu), "Atelectasis and Pleural Effusion heatmaps should differ"


def test_gradcam_trained_checkpoint_integration():
    """Verify Grad-CAM execution using the trained experimental model weights."""
    ckpt_path = Path("experiments/full_gpu/best_full_model.pt")
    if not ckpt_path.is_file():
        pytest.skip("Experimental checkpoint not found; skipping integration test.")

    model = DenseNet121MultiLabel(pretrained=False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)

    dummy_input = torch.randn(1, 1, 224, 224)
    for idx, disease in enumerate(TARGET_LABELS):
        hm = gradcam(model, dummy_input, idx)
        assert hm.shape == (224, 224)
        assert torch.isfinite(hm).all()
        assert hm.min() >= 0.0 and hm.max() <= 1.0
