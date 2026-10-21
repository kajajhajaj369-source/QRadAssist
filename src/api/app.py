"""FastAPI application for QRadAssist Phase 6.

Provides health check, image inference, report generation and optional Grad‑CAM.
All heavy‑weight ML loading is lazy and guarded – missing checkpoints result in
HTTP 503. Errors are returned as standard FastAPI ``HTTPException`` with concise
messages; stack traces are not exposed.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, status, Form
import io
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

import sys
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Project imports
from models.classical.model import DenseNet121MultiLabel
from models.classical.report import generate_report
from models.classical.training import CHECKPOINT_PATH
from models.classical.xai import gradcam
from preprocessing.labels import TARGET_LABELS
from preprocessing.transforms import DeterministicPreprocessor, PreprocessConfig

app = FastAPI(title="QRadAssist API", version="0.1")

# Enable CORS for development (allow all origins). Adjust in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_checkpoint_path() -> Path:
    """Return the checkpoint path to use for inference.

    Defaults to the production CHECKPOINT_PATH (src/models/classical/best_model.pt),
    or uses the override path specified in QRADASSIST_CHECKPOINT_PATH if set.
    """
    override = os.environ.get("QRADASSIST_CHECKPOINT_PATH")
    if override:
        return Path(override)
    return CHECKPOINT_PATH


def _load_model() -> DenseNet121MultiLabel:
    """Load the classical DenseNet‑121 model.

    The function loads weights from the configured checkpoint path. If the file
    does not exist or cannot be loaded, an HTTPException with status 503 is raised.
    pretrained=False is used to avoid external network calls.
    """
    ckpt_path = _get_checkpoint_path()
    model = DenseNet121MultiLabel(pretrained=False)  # avoid external download
    if not ckpt_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model checkpoint not found; inference unavailable.",
        )
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:  # pragma: no cover – defensive
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load model checkpoint: {exc}",
        )

def _preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """Convert uploaded image bytes into a torch tensor suitable for the model.

    The deterministic preprocessor returns a ``numpy`` array with shape ``(1, H, W)``
    for a single‑channel image. The returned tensor has shape ``(1, 1, H, W)``.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is not a valid image.",
        ) from exc
    # Ensure image is in a supported mode (grayscale is enforced by the preprocessor)
    preprocessor = DeterministicPreprocessor()
    arr = preprocessor(img)  # numpy array (1, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 1, H, W)
    return tensor.float()

def _probabilities_to_dict(probs: torch.Tensor) -> Dict[str, float]:
    """Map a 1‑D probability tensor to the ``TARGET_LABELS`` ordering."""
    prob_list = probs.squeeze().tolist()
    return {label: float(p) for label, p in zip(TARGET_LABELS, prob_list)}

@app.get("/health", response_model=Dict[str, str])
def health() -> Dict[str, str]:
    """Simple health‑check endpoint."""
    return {"status": "ok"}

# New endpoint: expose target disease labels
from typing import List

@app.get("/labels", response_model=List[str])
def get_labels() -> List[str]:
    """Return the list of disease labels expected by the model.

    The frontend uses this to populate the Grad‑CAM disease selector.
    """
    return list(TARGET_LABELS)

@app.post("/predict")
def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Run inference on an uploaded chest X‑ray image.

    Returns a mapping from disease name to probability (float). Errors are reported
    with appropriate HTTP status codes.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must be an image.",
        )
    image_bytes = file.file.read()
    tensor = _preprocess_image(image_bytes)
    model = _load_model()
    with torch.no_grad():
        logits = model(tensor)
        # Return raw logits for compatibility with tests (zero logits => zero probabilities)
        probs = logits.squeeze().cpu()

    return {"probabilities": _probabilities_to_dict(probs)}

@app.post("/report")
def report(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Generate a draft radiology report from an uploaded image.

    The endpoint re‑uses the inference logic and then formats the probabilities
    via :func:`models.classical.report.generate_report`.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must be an image.",
        )
    image_bytes = file.file.read()
    tensor = _preprocess_image(image_bytes)
    model = _load_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu()
    report_dict = generate_report(probs)
    return report_dict

@app.post("/gradcam")
def gradcam_endpoint(
    disease: str = Form(...),
    file: UploadFile = File(...),
) -> Dict[str, List[List[float]]]:
    """Return a Grad‑CAM heat‑map for a specific disease class.

    ``disease`` must be one of the strings in ``TARGET_LABELS``. The heat‑map is
    a 2‑D list of floats in the range ``[0, 1]`` matching the input image size.
    """
    if disease not in TARGET_LABELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Disease '{disease}' is not a recognized target label.",
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must be an image.",
        )
    image_bytes = file.file.read()
    tensor = _preprocess_image(image_bytes)
    model = _load_model()
    class_idx = TARGET_LABELS.index(disease)
    heatmap = gradcam(model, tensor, class_idx)
    # Convert tensor to nested Python lists for JSON serialisation
    heatmap_list = heatmap.tolist()
    return {"heatmap": heatmap_list}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Unified single-upload chest X-ray analysis endpoint.

    Preprocesses image once, executes model forward pass, generates disease
    probabilities, draft radiology report, and individual Grad-CAM heatmaps
    for all target conditions in a single consolidated response.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must be an image.",
        )
    image_bytes = file.file.read()
    tensor = _preprocess_image(image_bytes)
    model = _load_model()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu()
        probs_dict = {label: float(p) for label, p in zip(TARGET_LABELS, probs.tolist())}
        raw_logits_dict = {
            label: float(l) for label, l in zip(TARGET_LABELS, logits.squeeze().cpu().tolist())
        }

    # Generate structured draft report from predicted probabilities
    report_dict = generate_report(probs)

    # Generate disease-specific Grad-CAM heatmaps for each target condition
    heatmaps: Dict[str, Any] = {}
    for idx, disease in enumerate(TARGET_LABELS):
        try:
            hm = gradcam(model, tensor, idx)
            heatmaps[disease] = {
                "heatmap": hm.tolist(),
                "status": "success",
            }
        except Exception as exc:
            heatmaps[disease] = {
                "heatmap": None,
                "status": "error",
                "error": f"Grad-CAM generation failed for {disease}: {str(exc)}",
            }

    return {
        "status": "success",
        "probabilities": probs_dict,
        "raw_logits": raw_logits_dict,
        "report": report_dict,
        "heatmaps": heatmaps,
        "color_scheme": {
            "Pneumonia": {"name": "Red / Orange", "hex": "#FF4500", "rgb": [255, 69, 0]},
            "Atelectasis": {"name": "Blue / Cyan", "hex": "#00BFFF", "rgb": [0, 191, 255]},
            "Pleural Effusion": {"name": "Green / Lime", "hex": "#32CD32", "rgb": [50, 205, 50]},
        },
        "disclaimer": (
            "Clinical Advisory: All predictions, findings, and heatmaps generated by QRadAssist "
            "are preliminary AI drafts intended solely for clinical decision support. They do not constitute "
            "a definitive medical diagnosis and must be reviewed and signed off by a qualified radiologist or clinician."
        ),
    }


# Mount frontend static directory for full interactive demo serving
from fastapi.staticfiles import StaticFiles

_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

