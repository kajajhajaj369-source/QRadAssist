"""Report generation utilities for the classical multi‑label chest X‑ray model.

The module provides a lightweight, provider‑independent implementation that
converts disease probability predictions into a structured draft report.
It is deliberately simple and does **not** claim diagnostic certainty.
Grad‑CAM heatmaps can be attached as optional visual aids, but the generated
text does not infer causality from them.
"""

from __future__ import annotations

import textwrap
from typing import Mapping, Sequence, Tuple, Dict, Any

import torch
from torch import Tensor

from preprocessing.labels import TARGET_LABELS


def _format_findings(prob_dict: Mapping[str, float], threshold: float) -> list[str]:
    """Create human‑readable finding strings for diseases above *threshold*.

    Args:
        prob_dict: Mapping from disease name to predicted probability.
        threshold: Probability cutoff for reporting a positive finding.

    Returns:
        List of formatted strings, e.g. ``"Pneumonia (probability 0.82)"``.
    """
    findings = []
    for disease, prob in prob_dict.items():
        if prob >= threshold:
            findings.append(f"{disease} (probability {prob:.2f})")
    return findings


def _compose_impression(findings: Sequence[str]) -> str:
    """Compose a short impression paragraph from *findings*.

    If no findings are present, a generic "no acute disease" statement is used.
    """
    if not findings:
        return "No acute cardiopulmonary findings identified."
    joined = ", ".join(findings)
    return f"Findings suggest: {joined}."


def generate_report(
    prob_tensor: Tensor,
    *,
    threshold: float = 0.5,
    include_uncertain: bool = False,
    heatmaps: Mapping[str, Tensor] | None = None,
) -> Dict[str, Any]:
    """Generate a structured draft radiology report.

    Parameters
    ----------
    prob_tensor:
        Tensor of shape ``(C,)`` or ``(1, C)`` where ``C`` equals the number of
        target diseases (order given by ``preprocessing.labels.TARGET_LABELS``).
    threshold:
        Probability cutoff for treating a disease as a positive finding.
    include_uncertain:
        Placeholder for future handling of uncertain predictions; currently
        unused but kept for API stability.
    heatmaps:
        Optional mapping from disease name to a Grad‑CAM heat‑map tensor. The
        function does **not** embed images; it simply records that heat‑maps are
        available so downstream callers can attach them to the report.

    Returns
    -------
    dict
        ``{"findings": List[str], "impression": str, "heatmaps": dict}``
        where ``heatmaps`` mirrors the input mapping (or an empty dict).
    """
    if prob_tensor.ndim == 2 and prob_tensor.size(0) == 1:
        prob_tensor = prob_tensor.squeeze(0)
    if prob_tensor.ndim != 1 or prob_tensor.numel() != len(TARGET_LABELS):
        raise ValueError(
            f"prob_tensor must be a 1‑D tensor of length {len(TARGET_LABELS)} (got shape {tuple(prob_tensor.shape)})"
        )

    prob_dict = {disease: float(prob_tensor[i]) for i, disease in enumerate(TARGET_LABELS)}

    findings = _format_findings(prob_dict, threshold)
    impression = _compose_impression(findings)

    heatmap_dict: dict[str, Tensor] = {}
    if heatmaps:
        for disease, hm in heatmaps.items():
            if disease not in TARGET_LABELS:
                continue
            heatmap_dict[disease] = hm

    return {"findings": findings, "impression": impression, "heatmaps": heatmap_dict}
