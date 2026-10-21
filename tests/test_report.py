import torch
import pytest
from src.models.classical.report import generate_report
from preprocessing.labels import TARGET_LABELS


def test_generate_report_basic():
    # Create a probability tensor where first disease is above threshold
    probs = torch.tensor([0.8, 0.3, 0.6])
    report = generate_report(probs, threshold=0.5)
    # Findings should include Pneumonia and Pleural Effusion
    assert "Pneumonia (probability 0.80)" in report["findings"]
    assert "Pleural Effusion (probability 0.60)" in report["findings"]
    assert "Atelectasis" not in " ".join(report["findings"])
    # Impression should contain both findings
    assert "Findings suggest:" in report["impression"]
    assert "Pneumonia" in report["impression"]
    assert "Pleural Effusion" in report["impression"]


def test_generate_report_no_findings():
    probs = torch.zeros(len(TARGET_LABELS))
    report = generate_report(probs, threshold=0.5)
    assert report["findings"] == []
    assert report["impression"] == "No acute cardiopulmonary findings identified."


def test_generate_report_invalid_shape():
    probs = torch.randn(2, len(TARGET_LABELS))  # batch size >1 not allowed
    with pytest.raises(ValueError, match="prob_tensor must be a 1‑D tensor"):
        generate_report(probs)


def test_generate_report_with_heatmaps():
    probs = torch.tensor([0.4, 0.6, 0.2])
    dummy_heatmap = torch.randn(224, 224)
    heatmaps = {TARGET_LABELS[1]: dummy_heatmap}
    report = generate_report(probs, threshold=0.5, heatmaps=heatmaps)
    assert "Atelectasis" in report["heatmaps"]
    assert torch.equal(report["heatmaps"]["Atelectasis"], dummy_heatmap)
