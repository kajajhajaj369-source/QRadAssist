"""Initial data contracts for multi-label findings.

These structures are model-, XAI-, and QML-agnostic. They describe pipeline
outputs, not clinical truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


@dataclass
class DiseasePrediction:
    """One label in a multi-label result.

    `disease` is a registry id (for example, "pneumonia"), not a display name.
    `positive` is a pipeline decision flag, not a clinical diagnosis.
    """

    disease: str
    probability: float
    positive: bool

    def __post_init__(self) -> None:
        self.disease = _require_non_empty_string(self.disease, "disease")
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
            raise ValueError("probability must be a number between 0 and 1.")
        probability = float(self.probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1 inclusive.")
        if not isinstance(self.positive, bool):
            raise ValueError("positive must be a boolean.")
        self.probability = probability

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "probability": self.probability,
            "positive": self.positive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiseasePrediction:
        if not isinstance(data, dict):
            raise ValueError("A prediction must be an object.")
        return cls(
            disease=data.get("disease"),
            probability=data.get("probability"),
            positive=data.get("positive"),
        )


@dataclass
class XAIAttachment:
    """Optional explainability metadata attached to findings.

    The schema is method-agnostic. Extra method-specific fields belong in
    `details` so the prediction contract is not coupled to one XAI technique.
    """

    method: str
    target: Optional[str] = None
    heatmap_available: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.method = _require_non_empty_string(self.method, "xai.method")
        if self.target is not None:
            self.target = _require_non_empty_string(self.target, "xai.target")
        if not isinstance(self.heatmap_available, bool):
            raise ValueError("xai.heatmap_available must be a boolean.")
        if self.details is None:
            self.details = {}
        if not isinstance(self.details, dict):
            raise ValueError("xai.details must be an object if provided.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method,
            "heatmap_available": self.heatmap_available,
        }
        if self.target is not None:
            payload["target"] = self.target
        if self.details:
            payload["details"] = self.details
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XAIAttachment:
        if not isinstance(data, dict):
            raise ValueError("xai must be an object.")
        known = {"method", "target", "heatmap_available", "details"}
        extra = {key: value for key, value in data.items() if key not in known}
        details = dict(data.get("details") or {})
        details.update(extra)
        return cls(
            method=data.get("method"),
            target=data.get("target"),
            heatmap_available=data.get("heatmap_available", False),
            details=details,
        )


@dataclass
class AnalysisFindings:
    """Structured output of the image-analysis pipeline.

    Multiple predictions are required for multi-label analysis. XAI metadata
    is optional and may be omitted until an explanation method is selected.
    """

    predictions: list[DiseasePrediction]
    xai: list[XAIAttachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.predictions, list) or not self.predictions:
            raise ValueError("findings must include a non-empty predictions list.")
        normalized: list[DiseasePrediction] = []
        for item in self.predictions:
            if isinstance(item, DiseasePrediction):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(DiseasePrediction.from_dict(item))
            else:
                raise ValueError("Each prediction must be a DiseasePrediction or dict.")
        self.predictions = normalized

        disease_ids = [item.disease for item in self.predictions]
        duplicates = sorted({disease_id for disease_id in disease_ids if disease_ids.count(disease_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate disease ids in predictions: {duplicates}")

        if self.xai is None:
            self.xai = []
        if isinstance(self.xai, dict):
            self.xai = [XAIAttachment.from_dict(self.xai)]
        elif isinstance(self.xai, list):
            attachments: list[XAIAttachment] = []
            for item in self.xai:
                if isinstance(item, XAIAttachment):
                    attachments.append(item)
                elif isinstance(item, dict):
                    attachments.append(XAIAttachment.from_dict(item))
                else:
                    raise ValueError("Each xai entry must be an XAIAttachment or dict.")
            self.xai = attachments
        else:
            raise ValueError("xai must be omitted, an object, or a list of objects.")

        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object if provided.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "predictions": [item.to_dict() for item in self.predictions],
        }
        if len(self.xai) == 1:
            payload["xai"] = self.xai[0].to_dict()
        elif self.xai:
            payload["xai"] = [item.to_dict() for item in self.xai]
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisFindings:
        if not isinstance(data, dict):
            raise ValueError("Findings must be an object.")
        return cls(
            predictions=data.get("predictions"),
            xai=data.get("xai") or [],
            metadata=data.get("metadata") or {},
        )
