import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Import the FastAPI app
from src.api.app import app, _load_model, gradcam as original_gradcam, TARGET_LABELS

client = TestClient(app)

def _create_dummy_image_bytes(size=(224, 224)):
    # Create a simple gray image and return JPEG bytes
    from PIL import Image
    img = Image.new("L", size, color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# Helper to create a mock model that returns zero logits of appropriate size
def _mock_model(num_classes=len(TARGET_LABELS)):
    class MockModel:
        def __init__(self):
            self.backbone = SimpleNamespace(classifier=SimpleNamespace(out_features=num_classes),
                                            features=[None, None, None, None, None, None, None, None, None])
        def __call__(self, x):
            # Return zeros logits (batch, num_classes)
            import torch
            return torch.zeros((x.shape[0], num_classes))
        def eval(self):
            pass
        def to(self, device):
            return self
        def train(self):
            pass
    return MockModel()

# ---------------------------------------------------------------------------
# Health endpoint
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

# ---------------------------------------------------------------------------
# Predict – successful case
def test_predict_success(monkeypatch):
    # Patch model loader to return mock model
    monkeypatch.setattr('src.api.app._load_model', lambda: _mock_model())
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "probabilities" in data
    probs = data["probabilities"]
    # All probabilities should be 0.0 (sigmoid of 0)
    for label in TARGET_LABELS:
        assert label in probs
        assert abs(probs[label]) < 1e-6

# ---------------------------------------------------------------------------
# Predict – invalid image
def test_predict_invalid_image(monkeypatch):
    monkeypatch.setattr('src.api.app._load_model', lambda: _mock_model())
    files = {"file": ("not_image.txt", b"just some text", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "detail" in response.json()

# ---------------------------------------------------------------------------
# Predict – missing checkpoint (service unavailable)
def test_predict_missing_checkpoint(monkeypatch):
    # Force _load_model to raise the HTTPException as in real code
    from fastapi import HTTPException
    def raise_503():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model checkpoint not found; inference unavailable.")
    monkeypatch.setattr('src.api.app._load_model', raise_503)
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

# ---------------------------------------------------------------------------
# Report – successful case
def test_report_success(monkeypatch):
    monkeypatch.setattr('src.api.app._load_model', lambda: _mock_model())
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    response = client.post("/report", files=files)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # generate_report returns dict with keys findings, impression, heatmaps
    assert set(data.keys()) == {"findings", "impression", "heatmaps"}
    assert isinstance(data["findings"], list)
    assert isinstance(data["impression"], str)
    assert isinstance(data["heatmaps"], dict)

# ---------------------------------------------------------------------------
# GradCAM – successful case
def test_gradcam_success(monkeypatch):
    # Mock model loader and gradcam function
    monkeypatch.setattr('src.api.app._load_model', lambda: _mock_model())
    dummy_hm = SimpleNamespace()
    # Replace gradcam with stub returning a constant tensor
    def fake_gradcam(model, input_tensor, target_class, device=None):
        import torch
        # Return a 224x224 heatmap of 0.5 values
        return torch.full((224, 224), 0.5)
    monkeypatch.setattr('src.api.app.gradcam', fake_gradcam)
    img_bytes = _create_dummy_image_bytes()
    disease = TARGET_LABELS[0]
    files = {"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    response = client.post("/gradcam", data={"disease": disease}, files=files)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "heatmap" in data
    hm = data["heatmap"]
    # Should be a 2‑D list of floats (224x224) with value 0.5
    assert isinstance(hm, list)
    assert len(hm) == 224
    assert all(isinstance(row, list) and len(row) == 224 for row in hm)
    # Check a few values
    assert abs(hm[0][0] - 0.5) < 1e-6

# ---------------------------------------------------------------------------
# GradCAM – invalid disease label
def test_gradcam_invalid_disease(monkeypatch):
    monkeypatch.setattr('src.api.app._load_model', lambda: _mock_model())
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    response = client.post("/gradcam", data={"disease": "nonexistent"}, files=files)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "detail" in response.json()

# ---------------------------------------------------------------------------
# GradCAM – invalid image
def test_gradcam_invalid_image(monkeypatch):
    monkeypatch.setattr('src.api.app._load_model', lambda: _mock_model())
    files = {"file": ("not_image.txt", b"bad", "text/plain")}
    response = client.post("/gradcam", data={"disease": TARGET_LABELS[0]}, files=files)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "detail" in response.json()

# ---------------------------------------------------------------------------
# Checkpoint override resolution test
def test_checkpoint_path_resolution_env_override(monkeypatch):
    from src.api.app import _get_checkpoint_path, CHECKPOINT_PATH
    import os

    monkeypatch.delenv("QRADASSIST_CHECKPOINT_PATH", raising=False)
    assert _get_checkpoint_path() == CHECKPOINT_PATH

    custom_path = Path("experiments/full_gpu/best_full_model.pt")
    monkeypatch.setenv("QRADASSIST_CHECKPOINT_PATH", str(custom_path))
    assert _get_checkpoint_path() == custom_path

# ---------------------------------------------------------------------------
# End-to-end workflow test with experimental checkpoint
def test_end_to_end_workflow_experimental_model(monkeypatch):
    exp_ckpt = Path("experiments/full_gpu/best_full_model.pt")
    if not exp_ckpt.is_file():
        pytest.skip("Experimental checkpoint not found; skipping integration test.")

    monkeypatch.setenv("QRADASSIST_CHECKPOINT_PATH", str(exp_ckpt))
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("synthetic.jpg", img_bytes, "image/jpeg")}

    # 1. /predict
    resp_predict = client.post("/predict", files=files)
    assert resp_predict.status_code == status.HTTP_200_OK
    data_predict = resp_predict.json()
    assert "probabilities" in data_predict
    for label in TARGET_LABELS:
        assert label in data_predict["probabilities"]
        assert isinstance(data_predict["probabilities"][label], float)

    # 2. /report
    files = {"file": ("synthetic.jpg", img_bytes, "image/jpeg")}
    resp_report = client.post("/report", files=files)
    assert resp_report.status_code == status.HTTP_200_OK
    data_report = resp_report.json()
    assert set(data_report.keys()) == {"findings", "impression", "heatmaps"}
    assert isinstance(data_report["findings"], list)
    assert isinstance(data_report["impression"], str)

    # 3. /gradcam for each target disease
    for disease in TARGET_LABELS:
        files = {"file": ("synthetic.jpg", img_bytes, "image/jpeg")}
        resp_cam = client.post("/gradcam", data={"disease": disease}, files=files)
        assert resp_cam.status_code == status.HTTP_200_OK
        data_cam = resp_cam.json()
        assert "heatmap" in data_cam
        hm = data_cam["heatmap"]
        assert len(hm) == 224 and len(hm[0]) == 224


def test_frontend_static_serving():
    """Verify that root endpoint serves index.html and static assets are accessible."""
    resp = client.get("/")
    assert resp.status_code == status.HTTP_200_OK
    assert "<title>QRadAssist" in resp.text
    assert "Clinical Advisory" in resp.text

    resp_css = client.get("/style.css")
    assert resp_css.status_code == status.HTTP_200_OK

    resp_js = client.get("/script.js")
    assert resp_js.status_code == status.HTTP_200_OK


def test_analyze_success(monkeypatch):
    """Verify POST /analyze returns unified response with predictions, report, and heatmaps."""
    monkeypatch.setattr("src.api.app._load_model", lambda: _mock_model())
    monkeypatch.setattr(
        "src.api.app.gradcam",
        lambda model, tensor, class_idx: __import__("torch").zeros((224, 224)),
    )
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("xray.jpg", img_bytes, "image/jpeg")}
    response = client.post("/analyze", files=files)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["status"] == "success"
    assert "probabilities" in data
    assert "report" in data
    assert "heatmaps" in data
    assert "color_scheme" in data
    assert "disclaimer" in data

    # Check probabilities
    for label in TARGET_LABELS:
        assert label in data["probabilities"]
        assert 0.0 <= data["probabilities"][label] <= 1.0

    # Check report
    assert "findings" in data["report"]
    assert "impression" in data["report"]

    # Check heatmaps for all 3 classes
    for label in TARGET_LABELS:
        assert label in data["heatmaps"]
        hm_obj = data["heatmaps"][label]
        assert hm_obj["status"] == "success"
        hm = hm_obj["heatmap"]
        assert len(hm) == 224 and len(hm[0]) == 224

    # Check color scheme
    assert "Pneumonia" in data["color_scheme"]
    assert "Atelectasis" in data["color_scheme"]
    assert "Pleural Effusion" in data["color_scheme"]


def test_analyze_invalid_image():
    """Verify POST /analyze returns 422 for non-image uploads."""
    files = {"file": ("notes.txt", b"not an image", "text/plain")}
    response = client.post("/analyze", files=files)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_analyze_missing_checkpoint(monkeypatch):
    """Verify POST /analyze returns 503 when model checkpoint is missing."""
    monkeypatch.setattr(
        "src.api.app._get_checkpoint_path",
        lambda: Path("nonexistent_checkpoint.pt"),
    )
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("xray.jpg", img_bytes, "image/jpeg")}
    response = client.post("/analyze", files=files)
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_analyze_partial_heatmap_failure(monkeypatch):
    """Verify POST /analyze gracefully handles a failure in one disease's Grad-CAM."""
    monkeypatch.setattr("src.api.app._load_model", lambda: _mock_model())

    def fail_on_pneumonia(model, tensor, class_idx):
        if class_idx == TARGET_LABELS.index("Pneumonia"):
            raise RuntimeError("Synthetic Grad-CAM gradient hook error")
        return __import__("torch").zeros((224, 224))

    monkeypatch.setattr("src.api.app.gradcam", fail_on_pneumonia)
    img_bytes = _create_dummy_image_bytes()
    files = {"file": ("xray.jpg", img_bytes, "image/jpeg")}
    response = client.post("/analyze", files=files)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Pneumonia heatmap should report error
    assert data["heatmaps"]["Pneumonia"]["status"] == "error"
    assert "failed" in data["heatmaps"]["Pneumonia"]["error"].lower()

    # Atelectasis and Pleural Effusion should still succeed
    assert data["heatmaps"]["Atelectasis"]["status"] == "success"
    assert data["heatmaps"]["Pleural Effusion"]["status"] == "success"

    # Overall probabilities and report should still succeed
    assert "probabilities" in data
    assert "report" in data


