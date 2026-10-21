import os
import csv
import torch
import numpy as np
from pathlib import Path
from PIL import Image

import pytest

from src.models.classical.dataset import CheXpertDataset
from src.models.classical.model import DenseNet121MultiLabel
from src.models.classical.loss import masked_bce_with_logits
from src.models.classical.metrics import compute_auc_ap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "chexpert-small"
MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "test_manifest.csv"

@pytest.fixture(scope="module")
def dummy_image(tmp_path_factory):
    # Create a dummy 224x224 grayscale image
    img_path = tmp_path_factory.mktemp("imgs") / "dummy.jpg"
    img = Image.new("L", (224, 224), color=128)
    img.save(img_path)
    return img_path

@pytest.fixture(scope="module")
def manifest_file(dummy_image):
    # Write a simple manifest CSV with required columns
    manifest_path = MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "image_path",
                "raw_csv_path",
                "patient_id",
                "study_id",
                "view",
                "projection",
                "pneumonia_raw",
                "atelectasis_raw",
                "pleural_effusion_raw",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "image_path": str(dummy_image),
                "raw_csv_path": "",
                "patient_id": "1",
                "study_id": "1",
                "view": "Frontal",
                "projection": "AP",
                "pneumonia_raw": "1.0",
                "atelectasis_raw": "0.0",
                "pleural_effusion_raw": "-1.0",
                "split": "train",
            }
        )
    return manifest_path

def test_dataset_loading(manifest_file):
    from src.preprocessing.transforms import DeterministicPreprocessor
    preproc = DeterministicPreprocessor()
    ds = CheXpertDataset(manifest_file, preproc)
    assert len(ds) == 1
    sample = ds[0]
    # Image should be (1, 224, 224) float tensor
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (1, 224, 224)
    # Targets and mask should have length 3
    assert sample["target"].shape == (3,)
    assert sample["mask"].shape == (3,)
    # Check that uncertain label is masked out (pleural_effusion)
    assert sample["mask"][2].item() == 0.0

def test_loss_masking():
    logits = torch.tensor([[0.2, -0.5, 1.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss = masked_bce_with_logits(logits, targets, mask)
    # Compute expected loss manually for first two entries
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    expected = (bce(logits[:,0], targets[:,0]) + bce(logits[:,1], targets[:,1])) / 2.0
    assert torch.isclose(loss, expected)
    # All masked -> loss should be 0
    loss_all_masked = masked_bce_with_logits(logits, targets, torch.zeros_like(mask))
    assert loss_all_masked.item() == 0.0

def test_metrics_edge_cases():
    # Create logits and targets where classes 0 and 1 have only negatives,
    # and class 2 has both positives and a negative.
    logits = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.2, 0.3, 0.1]])
    targets = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 0]])  # class 0 and 1 only negatives; class 2 has 1 and 0
    mask = torch.ones_like(targets, dtype=torch.float32)
    auroc, ap = compute_auc_ap(logits, targets, mask)
    # AUROC for class 0 and 1 should be None (no positive samples)
    assert auroc["Pneumonia"] is None
    assert auroc["Atelectasis"] is None
    # AP should be 0.0 for classes with no positives
    assert ap["Pneumonia"] == 0.0
    assert ap["Atelectasis"] == 0.0
    # Class with positives should have valid scores
    assert auroc["Pleural Effusion"] is not None
    assert ap["Pleural Effusion"] is not None

def test_model_output_dimensions():
    model = DenseNet121MultiLabel(pretrained=False)
    dummy_input = torch.randn(2, 1, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 3)
